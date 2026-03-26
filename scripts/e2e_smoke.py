#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional


def log(message: str) -> None:
    print(f"[SMOKE] {message}")


def fail(message: str) -> int:
    print(f"[FAIL] {message}", file=sys.stderr)
    return 1


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class FakeKMClient:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def iter_document_content(self, km_id: str, version: str):
        del km_id, version
        yield self._payload


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.uploads: Dict[str, Dict[str, object]] = {}

    def upload_bytes(
        self,
        bucket: Optional[str],
        key: str,
        payload: bytes,
        expected_checksum: Optional[str] = None,
    ) -> str:
        from services.common.storage.object_store import compute_checksum

        checksum = compute_checksum(payload)
        if expected_checksum is not None and checksum != expected_checksum:
            raise AssertionError(f"checksum mismatch for {key}")
        self.uploads[key] = {
            "bucket": bucket,
            "payload": payload,
            "checksum": checksum,
        }
        return checksum


class DummyTextExtractor:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract(self, document, *, expected_checksum: Optional[str] = None):  # type: ignore[no-untyped-def]
        del expected_checksum
        from services.processing.stages import ExtractedDocument

        return ExtractedDocument(
            document_id=document.document_id,
            checksum=document.checksum,
            text=self._text,
            byte_size=len(self._text.encode("utf-8")),
            mime_type="text/plain",
            metadata={"km_id": document.km_id, "version": document.version},
        )


class StubEmbeddingExecutor:
    def execute(self, chunks):  # type: ignore[no-untyped-def]
        from services.processing.stages import EmbeddingExecutionResult

        embeddings = [[float(index)] for index, _chunk in enumerate(chunks)]
        sequence_indices = [chunk.sequence_index for chunk in chunks]
        return EmbeddingExecutionResult(embeddings=embeddings, sequence_indices=sequence_indices)


class RecordingChunkPersistence:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def persist(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        from services.processing.stages import ChunkPersistenceItem, ChunkPersistenceResult

        return ChunkPersistenceResult(
            chunk_key="chunks/smoke/chunks.jsonl",
            embedding_key="chunks/smoke/embeddings.jsonl",
            manifest_key="chunks/smoke/manifest.json",
            chunk_checksum="chunk-checksum",
            embedding_checksum="embedding-checksum",
            items=[
                ChunkPersistenceItem(
                    sequence_index=0,
                    token_count=8,
                    chunk_checksum="chunk-checksum",
                    embedding_checksum="embedding-checksum",
                )
            ],
        )


def configure_environment(temp_root: Path) -> None:
    os.environ.setdefault("OBJECT_STORE_ENDPOINT", "http://localhost:9000")
    os.environ.setdefault("OBJECT_STORE_ACCESS_KEY", "smoke-access")
    os.environ.setdefault("OBJECT_STORE_SECRET_KEY", "smoke-secret")
    os.environ.setdefault("OBJECT_STORE_BUCKET", "smoke-bucket")
    os.environ.setdefault("OBJECT_STORE_USE_SSL", "false")
    os.environ["INGESTION_DB_PATH"] = str(temp_root / "data" / "ingestion_jobs.db")
    os.environ["PROCESSING_DB_PATH"] = str(temp_root / "data" / "processing_jobs.db")
    os.environ["TESTSET_DB_PATH"] = str(temp_root / "data" / "testset_jobs.db")
    os.environ["EVAL_DB_PATH"] = str(temp_root / "data" / "eval_runs.db")
    os.environ["EVAL_OUTPUTS_DIR"] = str(temp_root / "data" / "eval_outputs")


def build_metric_payloads(aggregation_result) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    payloads: list[dict[str, object]] = []
    for name, distribution in aggregation_result.metrics.items():
        mean = distribution.average
        if mean >= 0.8:
            verdict = "ok"
        elif mean >= 0.5:
            verdict = "warn"
        else:
            verdict = "crit"
        payloads.append(
            {
                "name": name,
                "mean": distribution.average,
                "p50": distribution.p50,
                "p95": distribution.p95,
                "min": distribution.minimum,
                "max": distribution.maximum,
                "count": distribution.count,
                "verdict": verdict,
            }
        )
    return payloads


def assert_status(response, expected_status: int, context: str) -> Mapping[str, object]:  # type: ignore[no-untyped-def]
    if response.status_code != expected_status:
        raise AssertionError(f"{context} returned {response.status_code}: {response.text}")
    return response.json()


def run_smoke(root: Path) -> None:
    sys.path.insert(0, str(root))

    with tempfile.TemporaryDirectory(prefix="e2e-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        configure_environment(temp_root)

        with pushd(temp_root):
            from fastapi.testclient import TestClient

            from services.eval.context_capture import CapturedEvaluationItem
            from services.eval.rag_interface import RetrievedContext
            from services.ingestion.worker import IngestionWorker
            from services.processing.stages import ChunkBuilder
            from services.processing.worker import ProcessingWorker
            from services.testset.engine import TestsetGenerationEngine
            from services.testset.payloads import SourceChunk

            import services.ingestion.main as ingestion_main
            import services.processing.main as processing_main
            import services.testset.main as testset_main
            import services.eval.main as eval_main
            import services.reporting.main as reporting_main

            clients = [
                TestClient(ingestion_main.app),
                TestClient(processing_main.app),
                TestClient(testset_main.app),
                TestClient(eval_main.app),
                TestClient(reporting_main.app),
            ]

            try:
                ingestion_client, processing_client, testset_client, eval_client, reporting_client = clients

                log("Step 0: Health checks")
                for name, client in (
                    ("ingestion", ingestion_client),
                    ("processing", processing_client),
                    ("testset", testset_client),
                    ("eval", eval_client),
                    ("reporting", reporting_client),
                ):
                    body = assert_status(client.get("/health"), 200, f"{name} health")
                    if body.get("status") != "ok":
                        raise AssertionError(f"{name} health returned {body}")
                    log(f"  ✓ {name}")

                log("Step 1: Ingestion API + worker")
                ingestion_payload = {"km_id": "KM-SMOKE-001", "version": "v1"}
                ingestion_body = assert_status(
                    ingestion_client.post("/documents", json=ingestion_payload),
                    202,
                    "document submission",
                )
                ingestion_job_id = str(ingestion_body["job_id"])
                raw_document = b"Policy requires annual review of all compliance controls."
                ingestion_store = InMemoryObjectStore()
                ingestion_worker = IngestionWorker(
                    repository=ingestion_main.get_repository(),
                    km_client=FakeKMClient(raw_document),
                    object_store=ingestion_store,
                    bucket="smoke-bucket",
                )
                completed_ingestion = ingestion_worker.process_job(ingestion_job_id)
                if completed_ingestion.status != "completed" or not completed_ingestion.document_id:
                    raise AssertionError(f"ingestion did not complete: {completed_ingestion}")
                document_id = completed_ingestion.document_id
                log(f"  ✓ ingestion_job={ingestion_job_id} document_id={document_id}")

                log("Step 2: Processing API + worker")
                processing_payload = {"document_id": document_id, "profile_hash": "smoke-profile-v1"}
                processing_body = assert_status(
                    processing_client.post("/process-jobs", json=processing_payload),
                    202,
                    "processing submission",
                )
                processing_job_id = str(processing_body["job_id"])
                chunk_persistence = RecordingChunkPersistence()
                processing_worker = ProcessingWorker(
                    repository=processing_main.get_repository(),
                    document_repository=processing_main.get_document_repository(),
                    text_extractor=DummyTextExtractor(raw_document.decode("utf-8")),
                    chunk_builder=ChunkBuilder(),
                    embedding_executor=StubEmbeddingExecutor(),  # type: ignore[arg-type]
                    chunk_persistence=chunk_persistence,  # type: ignore[arg-type]
                )
                completed_processing = processing_worker.process_job(processing_job_id)
                if completed_processing.status != "completed":
                    raise AssertionError(f"processing did not complete: {completed_processing}")
                log(f"  ✓ processing_job={processing_job_id}")

                log("Step 3: Testset API + generation engine")
                testset_payload = {"method": "configurable", "max_total_samples": 2, "seed": 7}
                testset_body = assert_status(
                    testset_client.post("/testset-jobs", json=testset_payload),
                    202,
                    "testset submission",
                )
                testset_job_id = str(testset_body["job_id"])
                testset_store = InMemoryObjectStore()
                source_chunks = [
                    SourceChunk(
                        chunk_id="chunk-001",
                        document_id=document_id,
                        text=raw_document.decode("utf-8"),
                        metadata={"profile_hash": "smoke-profile-v1"},
                    )
                ]
                testset_engine = TestsetGenerationEngine(
                    repository=testset_main.get_repository(),
                    object_store=testset_store,
                    bucket="smoke-bucket",
                    storage_prefix="smoke-testsets",
                )
                testset_result = testset_engine.generate(job_id=testset_job_id, chunks=source_chunks)
                if testset_result.sample_count < 1:
                    raise AssertionError("testset generation produced no samples")
                log(f"  ✓ testset_job={testset_job_id} samples={testset_result.sample_count}")

                log("Step 4: Eval API + persistence pipeline")
                eval_payload = {"testset_id": testset_job_id, "profile": "baseline"}
                eval_body = assert_status(
                    eval_client.post("/eval-runs", json=eval_payload),
                    202,
                    "eval submission",
                )
                run_id = str(eval_body["run_id"])
                pipeline = eval_main.build_persistence_pipeline(run_id)
                metrics_records = [
                    {
                        "faithfulness": 0.92,
                        "answer_relevancy": 0.88,
                    },
                    {
                        "faithfulness": 0.86,
                        "answer_relevancy": 0.83,
                    },
                ]
                for index, metrics in enumerate(metrics_records, start=1):
                    item = CapturedEvaluationItem(
                        run_id=run_id,
                        sample_id=f"sample-{index:03d}",
                        question=f"What does sample {index} verify?",
                        answer="It verifies the smoke evaluation pipeline.",
                        contexts=(
                            RetrievedContext(
                                text=raw_document.decode("utf-8"),
                                document_id=document_id,
                                score=0.99,
                                metadata={"chunk_id": "chunk-001"},
                            ),
                        ),
                        success=True,
                        metadata={"profile": "baseline"},
                        raw={"source": "e2e_smoke"},
                    )
                    pipeline.submit(item, metrics)
                aggregation = pipeline.finalize()
                eval_repository = eval_main.get_repository()
                eval_repository.update_status(run_id, status="completed")
                run_record = eval_repository.get_run(run_id)
                if run_record is None or run_record.status != "completed":
                    raise AssertionError("evaluation run was not completed")
                for artifact_path in pipeline.artifacts().values():
                    if not artifact_path.exists():
                        raise AssertionError(f"missing eval artifact: {artifact_path}")
                log(f"  ✓ eval_run={run_id}")

                log("Step 5: Reporting API")
                metrics_payload = build_metric_payloads(aggregation)
                reporting_payload = {
                    "run_id": run_id,
                    "testset_id": testset_job_id,
                    "metrics_version": "1.0.0-smoke",
                    "evaluation_item_count": aggregation.counts["records"],
                    "metrics": metrics_payload,
                    "counts": dict(aggregation.counts),
                    "created_at": run_record.created_at,
                    "completed_at": run_record.updated_at,
                    "template": "executive",
                    "generate_pdf": False,
                }
                report_body = assert_status(
                    reporting_client.post("/reports", json=reporting_payload),
                    202,
                    "report submission",
                )
                if report_body.get("status") != "generating":
                    raise AssertionError(f"unexpected report response: {report_body}")
                reports = assert_status(reporting_client.get("/reports"), 200, "report listing")
                if not reports:
                    raise AssertionError("report listing returned no reports")
                html_response = reporting_client.get(f"/reports/{run_id}/executive/html")
                if html_response.status_code != 200:
                    raise AssertionError(f"html report retrieval failed: {html_response.text}")
                log(f"  ✓ report_run={run_id}")

                log("")
                log("E2E smoke test passed")
                log(f"  ingestion={ingestion_job_id}")
                log(f"  processing={processing_job_id}")
                log(f"  testset={testset_job_id}")
                log(f"  eval={run_id}")
                log(f"  report={run_id}")
            finally:
                for client in clients:
                    client.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        run_smoke(root)
    except Exception as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())