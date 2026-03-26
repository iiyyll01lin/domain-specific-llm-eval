from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pytest
from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers.testset_schema import TestsetSample as RagasTestsetSample
from jsonschema import Draft202012Validator

from services.common.errors import ServiceError
from services.common.events import EventPublisher
from services.common.storage.object_store import compute_checksum
from services.testset.engine import TestsetGenerationEngine
from services.testset.generator_core import GenerationStats
from services.testset.payloads import SourceChunk
from services.testset.repository import TestsetRepository


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.uploads: Dict[str, Dict[str, Any]] = {}

    def upload_bytes(
        self,
        bucket: str | None,
        key: str,
        payload: bytes,
        expected_checksum: str | None = None,
    ) -> str:
        checksum = compute_checksum(payload)
        if expected_checksum is not None:
            assert expected_checksum == checksum, "checksum mismatch"
        self.uploads[key] = {
            "bucket": bucket,
            "payload": payload,
            "checksum": checksum,
        }
        return checksum


class RecordingEventPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[Dict[str, Any]] = []
        super().__init__(transport=self._record)

    def _record(self, envelope: Dict[str, Any]) -> None:
        self.events.append(envelope)


class StubGenerator:
    def generate(
        self,
        *,
        chunks: Sequence[SourceChunk],
        config: Mapping[str, Any],
    ) -> tuple[Sequence[RagasTestsetSample], GenerationStats, Dict[str, Any]]:
        sample = RagasTestsetSample(
            eval_sample=SingleTurnSample(
                user_input="What is the policy?",
                reference="The policy mandates annual reviews.",
                reference_contexts=[chunks[0].text if chunks else ""],
                rubrics={"strategy": "baseline"},
            ),
            synthesizer_name=str(config.get("method", "unknown")),
        )
        stats = GenerationStats(
            generated=1,
            filtered=1,
            dropped_duplicates=0,
            dropped_length=0,
        )
        metadata = {
            "seed": config.get("seed", 0),
            "persona": {"role": "auditor"},
            "persona_count": 1,
            "scenarios": [
                {
                    "scenario_id": "scenario-1",
                    "label": "Annual review",
                    "instructions": "Review the compliance policy",
                }
            ],
            "scenario_count": 1,
            "strategies": ["baseline"],
        }
        return [sample], stats, metadata


class EmptyGenerator:
    def generate(
        self,
        *,
        chunks: Sequence[SourceChunk],
        config: Mapping[str, Any],
    ) -> tuple[Sequence[RagasTestsetSample], GenerationStats, Dict[str, Any]]:
        stats = GenerationStats(generated=0, filtered=0, dropped_duplicates=0, dropped_length=0)
        metadata = {"seed": config.get("seed", 0), "scenarios": [], "strategies": []}
        return [], stats, metadata


@pytest.fixture()
def temp_repo(tmp_path: Path) -> TestsetRepository:
    db_path = tmp_path / "test.db"
    return TestsetRepository(str(db_path))


def _create_chunk() -> SourceChunk:
    return SourceChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Compliance policies require annual review of processes.",
        metadata={"language": "en"},
    )


def test_generate_persists_artifacts_and_metadata(temp_repo: TestsetRepository, tmp_path: Path) -> None:
    job = temp_repo.create_job(
        config_hash="hash-123",
        config={"method": "baseline", "seed": 99, "max_total_samples": 1},
    )
    store = InMemoryObjectStore()
    events = RecordingEventPublisher()
    engine = TestsetGenerationEngine(
        repository=temp_repo,
        object_store=store,
        generator=StubGenerator(),
        event_publisher=events,
        storage_prefix="unit-test/",
        bucket="test-bucket",
    )

    result = engine.generate(job_id=job.job_id, chunks=[_create_chunk()])

    assert result.sample_count == 1
    assert set(result.artifact_paths.keys()) == {"samples", "metadata", "personas", "scenarios"}
    assert result.metadata["job_id"] == job.job_id

    stored_job = temp_repo.get_job(job.job_id)
    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.sample_count == 1
    assert stored_job.persona_count == 1
    assert stored_job.scenario_count == 1
    assert stored_job.artifact_prefix == f"unit-test/{job.job_id}"

    samples_key = stored_job.artifact_paths["samples"]
    metadata_key = stored_job.artifact_paths["metadata"]
    personas_key = stored_job.artifact_paths["personas"]
    scenarios_key = stored_job.artifact_paths["scenarios"]
    assert samples_key in store.uploads
    assert metadata_key in store.uploads
    assert personas_key in store.uploads
    assert scenarios_key in store.uploads

    samples_payload = store.uploads[samples_key]["payload"].decode("utf-8").strip()
    sample_record = json.loads(samples_payload)
    assert sample_record["synthesizer_name"] == "baseline"

    metadata_payload = json.loads(store.uploads[metadata_key]["payload"].decode("utf-8"))
    assert metadata_payload["persona_count"] == 1
    assert metadata_payload["scenario_count"] == 1
    assert metadata_payload["checksum"] == store.uploads[samples_key]["checksum"]
    assert metadata_payload["personas_artifact"] == personas_key
    assert metadata_payload["scenarios_artifact"] == scenarios_key

    personas_payload = json.loads(store.uploads[personas_key]["payload"].decode("utf-8"))
    assert personas_payload["count"] == 1
    assert personas_payload["items"][0]["role"] == "auditor"

    scenarios_payload = json.loads(store.uploads[scenarios_key]["payload"].decode("utf-8"))
    assert scenarios_payload["count"] == 1
    assert scenarios_payload["items"][0]["scenario_id"] == "scenario-1"

    assert len(events.events) == 1
    envelope = events.events[0]
    schema_path = Path(__file__).resolve().parents[3] / "events" / "schemas" / "testset.created.v1.json"
    schema = json.loads(schema_path.read_text("utf-8"))
    Draft202012Validator(schema).validate(envelope)
    assert envelope["event"] == "testset.created"
    payload = envelope["payload"]
    assert payload["testset_id"] == job.job_id
    assert payload["sample_count"] == 1
    assert payload["config_hash"] == job.config_hash
    assert payload["seed"] == 99


def test_generate_requires_existing_job(temp_repo: TestsetRepository) -> None:
    engine = TestsetGenerationEngine(
        repository=temp_repo,
        object_store=InMemoryObjectStore(),
        generator=StubGenerator(),
        event_publisher=RecordingEventPublisher(),
    )

    with pytest.raises(ServiceError) as exc:
        engine.generate(job_id="missing", chunks=[_create_chunk()])

    assert exc.value.error_code == "testset_job_missing"


def test_generate_marks_failure_when_empty(temp_repo: TestsetRepository) -> None:
    job = temp_repo.create_job(
        config_hash="hash-empty",
        config={"method": "baseline", "seed": 101, "max_total_samples": 1},
    )
    engine = TestsetGenerationEngine(
        repository=temp_repo,
        object_store=InMemoryObjectStore(),
        generator=EmptyGenerator(),
        event_publisher=RecordingEventPublisher(),
    )

    with pytest.raises(ServiceError) as exc:
        engine.generate(job_id=job.job_id, chunks=[_create_chunk()])

    assert exc.value.error_code == "generation_empty"

    stored_job = temp_repo.get_job(job.job_id)
    assert stored_job is not None
    assert stored_job.status == "error"
    assert stored_job.error_code == "generation_empty"