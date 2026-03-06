#!/usr/bin/env bash
# TASK-101: E2E Pipeline Smoke Test
# Runs a minimal doc→ingestion→processing→testset→eval→report chain and
# verifies that all artefacts are produced.
#
# Usage:
#   ./scripts/e2e_smoke.sh [--base-url http://localhost:8001]
#
# Requirements:
#   - All services running (docker-compose or local)
#   - curl, jq available
#
# Exit codes:
#   0 - smoke test passed
#   1 - one or more steps failed
set -euo pipefail

BASE_INGESTION="${BASE_INGESTION:-http://localhost:8001}"
BASE_PROCESSING="${BASE_PROCESSING:-http://localhost:8002}"
BASE_TESTSET="${BASE_TESTSET:-http://localhost:8003}"
BASE_EVAL="${BASE_EVAL:-http://localhost:8004}"
BASE_REPORTING="${BASE_REPORTING:-http://localhost:8005}"
TIMEOUT="${POLL_TIMEOUT:-120}"      # seconds to wait for each job
POLL_INTERVAL=3

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SMOKE]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 0 – health checks
# ---------------------------------------------------------------------------
log "Step 0: Health checks"
for svc_url in "$BASE_INGESTION" "$BASE_PROCESSING" "$BASE_TESTSET" "$BASE_EVAL" "$BASE_REPORTING"; do
  status=$(curl -sf "$svc_url/health" | jq -r '.status' 2>/dev/null || echo "error")
  if [[ "$status" != "ok" ]]; then
    fail "Service at $svc_url is not healthy (got '$status')"
  fi
  log "  ✓ $svc_url"
done

# ---------------------------------------------------------------------------
# Step 1 – ingest a minimal CSV document
# ---------------------------------------------------------------------------
log "Step 1: Ingest document"
INGEST_PAYLOAD='{"source_uri":"smoke://test.csv","metadata":{"title":"smoke test"}}'
INGEST_RESP=$(curl -sf -X POST "$BASE_INGESTION/ingestion-jobs" \
  -H "Content-Type: application/json" \
  -d "$INGEST_PAYLOAD")
INGEST_JOB_ID=$(echo "$INGEST_RESP" | jq -r '.job_id')
[[ "$INGEST_JOB_ID" != "null" && -n "$INGEST_JOB_ID" ]] || fail "Ingestion job creation failed: $INGEST_RESP"
log "  Ingestion job: $INGEST_JOB_ID"

# Poll until completed
elapsed=0
while true; do
  STATUS=$(curl -sf "$BASE_INGESTION/ingestion-jobs/$INGEST_JOB_ID" | jq -r '.status')
  [[ "$STATUS" == "completed" ]] && break
  [[ "$STATUS" == "failed" ]]    && fail "Ingestion job failed"
  sleep $POLL_INTERVAL
  elapsed=$((elapsed + POLL_INTERVAL))
  [[ $elapsed -ge $TIMEOUT ]]    && fail "Ingestion job timed out after ${TIMEOUT}s"
done
log "  ✓ Ingestion completed"

# ---------------------------------------------------------------------------
# Step 2 – processing
# ---------------------------------------------------------------------------
log "Step 2: Processing"
PROC_PAYLOAD="{\"ingestion_job_id\":\"$INGEST_JOB_ID\"}"
PROC_RESP=$(curl -sf -X POST "$BASE_PROCESSING/processing-jobs" \
  -H "Content-Type: application/json" -d "$PROC_PAYLOAD")
PROC_JOB_ID=$(echo "$PROC_RESP" | jq -r '.job_id')
[[ "$PROC_JOB_ID" != "null" && -n "$PROC_JOB_ID" ]] || fail "Processing job creation failed"
log "  Processing job: $PROC_JOB_ID"

elapsed=0
while true; do
  STATUS=$(curl -sf "$BASE_PROCESSING/processing-jobs/$PROC_JOB_ID" | jq -r '.status')
  [[ "$STATUS" == "completed" ]] && break
  [[ "$STATUS" == "failed" ]]    && fail "Processing job failed"
  sleep $POLL_INTERVAL
  elapsed=$((elapsed + POLL_INTERVAL))
  [[ $elapsed -ge $TIMEOUT ]]    && fail "Processing job timed out"
done
log "  ✓ Processing completed"

# ---------------------------------------------------------------------------
# Step 3 – testset generation
# ---------------------------------------------------------------------------
log "Step 3: Testset generation"
TS_PAYLOAD="{\"processing_job_id\":\"$PROC_JOB_ID\",\"max_samples\":2}"
TS_RESP=$(curl -sf -X POST "$BASE_TESTSET/testset-jobs" \
  -H "Content-Type: application/json" -d "$TS_PAYLOAD")
TS_JOB_ID=$(echo "$TS_RESP" | jq -r '.job_id')
[[ "$TS_JOB_ID" != "null" && -n "$TS_JOB_ID" ]] || fail "Testset job creation failed"
log "  Testset job: $TS_JOB_ID"

elapsed=0
while true; do
  STATUS=$(curl -sf "$BASE_TESTSET/testset-jobs/$TS_JOB_ID" | jq -r '.status')
  [[ "$STATUS" == "completed" ]] && break
  [[ "$STATUS" == "failed" ]]    && fail "Testset job failed"
  sleep $POLL_INTERVAL
  elapsed=$((elapsed + POLL_INTERVAL))
  [[ $elapsed -ge $TIMEOUT ]]    && fail "Testset job timed out"
done
log "  ✓ Testset completed"

# ---------------------------------------------------------------------------
# Step 4 – evaluation
# ---------------------------------------------------------------------------
log "Step 4: Evaluation"
EVAL_PAYLOAD="{\"testset_job_id\":\"$TS_JOB_ID\"}"
EVAL_RESP=$(curl -sf -X POST "$BASE_EVAL/eval-runs" \
  -H "Content-Type: application/json" -d "$EVAL_PAYLOAD")
EVAL_RUN_ID=$(echo "$EVAL_RESP" | jq -r '.run_id')
[[ "$EVAL_RUN_ID" != "null" && -n "$EVAL_RUN_ID" ]] || fail "Eval run creation failed"
log "  Eval run: $EVAL_RUN_ID"

elapsed=0
while true; do
  STATUS=$(curl -sf "$BASE_EVAL/eval-runs/$EVAL_RUN_ID" | jq -r '.status')
  [[ "$STATUS" == "completed" ]] && break
  [[ "$STATUS" == "failed" ]]    && fail "Eval run failed"
  sleep $POLL_INTERVAL
  elapsed=$((elapsed + POLL_INTERVAL))
  [[ $elapsed -ge $TIMEOUT ]]    && fail "Eval run timed out"
done
log "  ✓ Evaluation completed"

# ---------------------------------------------------------------------------
# Step 5 – report
# ---------------------------------------------------------------------------
log "Step 5: Reporting"
RPT_PAYLOAD="{\"eval_run_id\":\"$EVAL_RUN_ID\"}"
RPT_RESP=$(curl -sf -X POST "$BASE_REPORTING/reports" \
  -H "Content-Type: application/json" -d "$RPT_PAYLOAD")
REPORT_ID=$(echo "$RPT_RESP" | jq -r '.report_id')
[[ "$REPORT_ID" != "null" && -n "$REPORT_ID" ]] || fail "Report creation failed"
log "  ✓ Report: $REPORT_ID"

# ---------------------------------------------------------------------------
log ""
log "╔══════════════════════════════╗"
log "║  E2E Smoke Test PASSED ✓     ║"
log "╚══════════════════════════════╝"
log "  ingestion=$INGEST_JOB_ID"
log "  processing=$PROC_JOB_ID"
log "  testset=$TS_JOB_ID"
log "  eval=$EVAL_RUN_ID"
log "  report=$REPORT_ID"
