#!/usr/bin/env bash
# TASK-101: E2E Pipeline Smoke Test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.services.yml}"
TEMP_ENV="$(mktemp)"
RAW_DOCUMENT='Policy requires annual review of all compliance controls.'
SMOKE_USE_PREBUILT_IMAGE="${SMOKE_USE_PREBUILT_IMAGE:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SMOKE]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

cleanup() {
	COMPOSE_ENV_FILE="$TEMP_ENV" docker compose --env-file "$TEMP_ENV" -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
  rm -f "$TEMP_ENV"
}

trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

compose() {
	COMPOSE_ENV_FILE="$TEMP_ENV" docker compose --env-file "$TEMP_ENV" -f "$COMPOSE_FILE" "$@"
}

wait_http() {
  local url="$1"
  local label="$2"
  local timeout="${3:-120}"
  local elapsed=0
  while ! curl -fsS "$url" >/dev/null 2>&1; do
	sleep 2
	elapsed=$((elapsed + 2))
	if [[ "$elapsed" -ge "$timeout" ]]; then
	  fail "$label did not become ready within ${timeout}s"
	fi
  done
}

json_field() {
  local json_payload="$1"
  local json_expr="$2"
  JSON_PAYLOAD="$json_payload" JSON_EXPR="$json_expr" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["JSON_PAYLOAD"])
expr = os.environ["JSON_EXPR"]

if expr == "length":
	value = len(payload)
else:
	if not expr.startswith("."):
		raise SystemExit(f"unsupported expression: {expr}")
	value = payload
	for part in expr.lstrip(".").split("."):
		if not part:
			continue
		value = value[part]

if value is None:
	print("null")
elif isinstance(value, bool):
	print("true" if value else "false")
else:
	print(value)
PY
}

cat > "$TEMP_ENV" <<EOF
PYTHONPATH=/app
LOG_LEVEL=INFO
OTEL_EXPORTER_OTLP_ENDPOINT=
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
MODELS_CACHE_PATH=/var/cache/rag-models
EXTENSIONS_DIR=/extensions
OBJECT_STORE_ENDPOINT=http://minio:9000
OBJECT_STORE_REGION=us-east-1
OBJECT_STORE_ACCESS_KEY=minioadmin
OBJECT_STORE_SECRET_KEY=minioadmin123
OBJECT_STORE_BUCKET=rag-eval-smoke
OBJECT_STORE_USE_SSL=false
OBJECT_STORE_MAX_ATTEMPTS=3
OBJECT_STORE_BACKOFF_SECONDS=0.5
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=rag-eval-smoke
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-}
PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-}
PIP_NETWORK_CHECK_URL=${PIP_NETWORK_CHECK_URL:-https://pypi.org/simple/}
PIP_NETWORK_TIMEOUT=${PIP_NETWORK_TIMEOUT:-5}
SERVICE_IMAGE_NAME=${SERVICE_IMAGE_NAME:-rag-eval}
SERVICE_IMAGE_TAG=${SERVICE_IMAGE_TAG:-dev}
ENABLE_GPU=${ENABLE_GPU:-false}
EOF

require_cmd docker
require_cmd curl

log "Step 0: Start compose-backed services"
compose up -d minio
wait_http "http://localhost:9000/minio/health/live" "MinIO"
compose run --rm minio-init >/dev/null
if [[ "$SMOKE_USE_PREBUILT_IMAGE" == "1" ]]; then
	log "  ↳ using prebuilt service image ${SERVICE_IMAGE_NAME:-rag-eval}:${SERVICE_IMAGE_TAG:-dev}"
	compose pull ingestion processing testset eval reporting >/dev/null
	compose up -d --no-build ingestion processing testset eval reporting
else
	log "  ↳ building service image via compose"
	if ! compose up -d --build ingestion processing testset eval reporting; then
		fail "Compose image build failed. This environment cannot reach PyPI during Docker builds. Set PIP_INDEX_URL/PIP_EXTRA_INDEX_URL/PIP_TRUSTED_HOST to an internal mirror, or rerun with SMOKE_USE_PREBUILT_IMAGE=1 plus SERVICE_IMAGE_NAME/SERVICE_IMAGE_TAG for a prebuilt image."
	fi
fi

for svc_url in \
  "http://localhost:8001/health" \
  "http://localhost:8002/health" \
  "http://localhost:8003/health" \
  "http://localhost:8004/health" \
  "http://localhost:8005/health"; do
  wait_http "$svc_url" "$svc_url"
done
log "  ✓ services healthy"

log "Step 1: Submit ingestion job over HTTP"
INGEST_RESP="$(curl -fsS -X POST http://localhost:8001/documents -H 'Content-Type: application/json' -d '{"km_id":"KM-SMOKE-001","version":"v1"}')"
INGEST_JOB_ID="$(json_field "$INGEST_RESP" '.job_id')"
[[ -n "$INGEST_JOB_ID" && "$INGEST_JOB_ID" != "null" ]] || fail "Ingestion submission failed: $INGEST_RESP"

INGEST_RESULT="$(compose exec -T -e SMOKE_INGEST_JOB_ID="$INGEST_JOB_ID" -e SMOKE_DOCUMENT_TEXT="$RAW_DOCUMENT" ingestion python - <<'PY'
import json
import os

from services.common.storage.object_store import ObjectStoreClient
from services.ingestion.main import get_repository
from services.ingestion.worker import IngestionWorker


class FakeKMClient:
	def __init__(self, payload: str) -> None:
		self._payload = payload.encode("utf-8")

	def iter_document_content(self, km_id: str, version: str):
		del km_id, version
		yield self._payload


worker = IngestionWorker(
	repository=get_repository(),
	km_client=FakeKMClient(os.environ["SMOKE_DOCUMENT_TEXT"]),
	object_store=ObjectStoreClient(),
)
job = worker.process_job(os.environ["SMOKE_INGEST_JOB_ID"])
print(json.dumps({"status": job.status, "document_id": job.document_id}))
PY
)"
INGEST_STATUS="$(json_field "$INGEST_RESULT" '.status')"
DOCUMENT_ID="$(json_field "$INGEST_RESULT" '.document_id')"
[[ "$INGEST_STATUS" == "completed" && -n "$DOCUMENT_ID" && "$DOCUMENT_ID" != "null" ]] || fail "Ingestion worker failed: $INGEST_RESULT"
log "  ✓ ingestion_job=$INGEST_JOB_ID document_id=$DOCUMENT_ID"

log "Step 2: Submit processing job over HTTP"
PROCESS_RESP="$(curl -fsS -X POST http://localhost:8002/process-jobs -H 'Content-Type: application/json' -d "{\"document_id\":\"$DOCUMENT_ID\",\"profile_hash\":\"smoke-profile-v1\"}")"
PROCESS_JOB_ID="$(json_field "$PROCESS_RESP" '.job_id')"
[[ -n "$PROCESS_JOB_ID" && "$PROCESS_JOB_ID" != "null" ]] || fail "Processing submission failed: $PROCESS_RESP"

PROCESS_RESULT="$(compose exec -T -e SMOKE_PROCESS_JOB_ID="$PROCESS_JOB_ID" processing python - <<'PY'
import json
import os

from services.common.storage.object_store import ObjectStoreClient
from services.processing.main import get_document_repository, get_repository
from services.processing.stages import ChunkBuilder, ChunkPersistence, EmbeddingBatchExecutor, TextExtractor
from services.processing.worker import ProcessingWorker


class StubProvider:
	def embed(self, texts, *, timeout):
		del timeout
		return [[float(index)] for index, _ in enumerate(texts)]


object_store = ObjectStoreClient()
worker = ProcessingWorker(
	repository=get_repository(),
	document_repository=get_document_repository(),
	text_extractor=TextExtractor(object_store),
	chunk_builder=ChunkBuilder(),
	embedding_executor=EmbeddingBatchExecutor(StubProvider()),
	chunk_persistence=ChunkPersistence(object_store),
)
job = worker.process_job(os.environ["SMOKE_PROCESS_JOB_ID"])
print(json.dumps({"status": job.status, "job_id": job.job_id}))
PY
)"
PROCESS_STATUS="$(json_field "$PROCESS_RESULT" '.status')"
[[ "$PROCESS_STATUS" == "completed" ]] || fail "Processing worker failed: $PROCESS_RESULT"
log "  ✓ processing_job=$PROCESS_JOB_ID"

log "Step 3: Submit testset job over HTTP"
TESTSET_RESP="$(curl -fsS -X POST http://localhost:8003/testset-jobs -H 'Content-Type: application/json' -d '{"method":"configurable","max_total_samples":2,"seed":7}')"
TESTSET_JOB_ID="$(json_field "$TESTSET_RESP" '.job_id')"
[[ -n "$TESTSET_JOB_ID" && "$TESTSET_JOB_ID" != "null" ]] || fail "Testset submission failed: $TESTSET_RESP"

TESTSET_RESULT="$(compose exec -T -e SMOKE_TESTSET_JOB_ID="$TESTSET_JOB_ID" -e SMOKE_DOCUMENT_ID="$DOCUMENT_ID" -e SMOKE_DOCUMENT_TEXT="$RAW_DOCUMENT" testset python - <<'PY'
import json
import os

from services.common.storage.object_store import ObjectStoreClient
from services.testset.engine import TestsetGenerationEngine
from services.testset.main import get_repository
from services.testset.payloads import SourceChunk


engine = TestsetGenerationEngine(
	repository=get_repository(),
	object_store=ObjectStoreClient(),
	storage_prefix="smoke-testsets",
)
result = engine.generate(
	job_id=os.environ["SMOKE_TESTSET_JOB_ID"],
	chunks=[
		SourceChunk(
			chunk_id="chunk-001",
			document_id=os.environ["SMOKE_DOCUMENT_ID"],
			text=os.environ["SMOKE_DOCUMENT_TEXT"],
			metadata={"profile_hash": "smoke-profile-v1"},
		)
	],
)
job = get_repository().get_job(os.environ["SMOKE_TESTSET_JOB_ID"])
print(json.dumps({"status": job.status if job else None, "sample_count": result.sample_count}))
PY
)"
TESTSET_STATUS="$(json_field "$TESTSET_RESULT" '.status')"
TESTSET_SAMPLE_COUNT="$(json_field "$TESTSET_RESULT" '.sample_count')"
[[ "$TESTSET_STATUS" == "completed" ]] || fail "Testset generation failed: $TESTSET_RESULT"
log "  ✓ testset_job=$TESTSET_JOB_ID samples=$TESTSET_SAMPLE_COUNT"

log "Step 4: Submit evaluation run over HTTP"
EVAL_RESP="$(curl -fsS -X POST http://localhost:8004/eval-runs -H 'Content-Type: application/json' -d "{\"testset_id\":\"$TESTSET_JOB_ID\",\"profile\":\"baseline\"}")"
EVAL_RUN_ID="$(json_field "$EVAL_RESP" '.run_id')"
[[ -n "$EVAL_RUN_ID" && "$EVAL_RUN_ID" != "null" ]] || fail "Eval submission failed: $EVAL_RESP"

EVAL_RESULT="$(compose exec -T -e SMOKE_RUN_ID="$EVAL_RUN_ID" -e SMOKE_DOCUMENT_ID="$DOCUMENT_ID" -e SMOKE_DOCUMENT_TEXT="$RAW_DOCUMENT" eval python - <<'PY'
import json
import os

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.main import build_persistence_pipeline, get_repository
from services.eval.rag_interface import RetrievedContext


def metric_payload(name, distribution):
	mean = distribution.average
	verdict = "ok" if mean >= 0.8 else "warn" if mean >= 0.5 else "crit"
	return {
		"name": name,
		"mean": distribution.average,
		"p50": distribution.p50,
		"p95": distribution.p95,
		"min": distribution.minimum,
		"max": distribution.maximum,
		"count": distribution.count,
		"verdict": verdict,
	}


run_id = os.environ["SMOKE_RUN_ID"]
document_id = os.environ["SMOKE_DOCUMENT_ID"]
document_text = os.environ["SMOKE_DOCUMENT_TEXT"]
pipeline = build_persistence_pipeline(run_id)
for index, metrics in enumerate((
	{"faithfulness": 0.92, "answer_relevancy": 0.88},
	{"faithfulness": 0.86, "answer_relevancy": 0.83},
), start=1):
	item = CapturedEvaluationItem(
		run_id=run_id,
		sample_id=f"sample-{index:03d}",
		question=f"What does sample {index} verify?",
		answer="It verifies the deployed smoke pipeline.",
		contexts=(
			RetrievedContext(
				text=document_text,
				document_id=document_id,
				score=0.99,
				metadata={"chunk_id": "chunk-001"},
			),
		),
		success=True,
		metadata={"profile": "baseline"},
		raw={"source": "compose_e2e_smoke"},
	)
	pipeline.submit(item, metrics)

aggregation = pipeline.finalize()
repository = get_repository()
repository.update_status(run_id, status="completed")
run = repository.get_run(run_id)
print(json.dumps({
	"status": run.status if run else None,
	"created_at": run.created_at if run else None,
	"completed_at": run.updated_at if run else None,
	"evaluation_item_count": aggregation.counts["records"],
	"counts": dict(aggregation.counts),
	"metrics": [metric_payload(name, distribution) for name, distribution in aggregation.metrics.items()],
}))
PY
)"
EVAL_STATUS="$(json_field "$EVAL_RESULT" '.status')"
[[ "$EVAL_STATUS" == "completed" ]] || fail "Eval execution failed: $EVAL_RESULT"
log "  ✓ eval_run=$EVAL_RUN_ID"

log "Step 5: Generate report over HTTP"
REPORT_PAYLOAD="$(python3 - <<PY
import json

eval_result = json.loads('''$EVAL_RESULT''')
payload = {
	"run_id": "$EVAL_RUN_ID",
	"testset_id": "$TESTSET_JOB_ID",
	"metrics_version": "1.0.0-smoke",
	"evaluation_item_count": eval_result["evaluation_item_count"],
	"metrics": eval_result["metrics"],
	"counts": eval_result["counts"],
	"created_at": eval_result["created_at"],
	"completed_at": eval_result["completed_at"],
	"template": "executive",
	"generate_pdf": False,
}
print(json.dumps(payload))
PY
)"
REPORT_RESP="$(curl -fsS -X POST http://localhost:8005/reports -H 'Content-Type: application/json' -d "$REPORT_PAYLOAD")"
REPORT_STATUS="$(json_field "$REPORT_RESP" '.status')"
[[ "$REPORT_STATUS" == "generating" ]] || fail "Reporting submission failed: $REPORT_RESP"

REPORT_LIST="$(curl -fsS http://localhost:8005/reports)"
REPORT_COUNT="$(json_field "$REPORT_LIST" 'length')"
[[ "$REPORT_COUNT" -ge 1 ]] || fail "No reports returned from reporting service"

curl -fsS "http://localhost:8005/reports/$EVAL_RUN_ID/executive/html" >/dev/null || fail "HTML report retrieval failed"
log "  ✓ report_run=$EVAL_RUN_ID"

log ""
log "Compose-backed E2E smoke test passed"
log "  ingestion=$INGEST_JOB_ID"
log "  processing=$PROCESS_JOB_ID"
log "  testset=$TESTSET_JOB_ID"
log "  eval=$EVAL_RUN_ID"
log "  report=$EVAL_RUN_ID"
