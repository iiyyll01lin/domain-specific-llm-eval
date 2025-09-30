from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

from services.common.errors import ServiceError
from services.common.storage.object_store import ObjectStoreClient, compute_checksum
from services.testset.generator_core import GenerationStats, GeneratorCore
from services.testset.payloads import SourceChunk
from services.testset.repository import TestsetRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestsetGenerationResult:
    sample_count: int
    artifact_paths: Dict[str, str]
    stats: GenerationStats
    metadata: Dict[str, Any]


class TestsetGenerationEngine:
    """Coordinates deterministic testset generation and artifact persistence."""

    def __init__(
        self,
        repository: TestsetRepository,
        object_store: ObjectStoreClient,
        *,
        generator: Optional[GeneratorCore] = None,
        storage_prefix: str = "testsets",
        bucket: Optional[str] = None,
    ) -> None:
        if not storage_prefix.strip():
            raise ValueError("storage_prefix must be non-empty")
        self._repository = repository
        self._object_store = object_store
        self._generator = generator or GeneratorCore()
        self._storage_prefix = storage_prefix.strip("/")
        self._bucket = bucket

    def generate(
        self,
        *,
        job_id: str,
        chunks: Sequence[SourceChunk],
    ) -> TestsetGenerationResult:
        job = self._repository.get_job(job_id)
        if job is None:
            raise ServiceError(
                error_code="testset_job_missing",
                message=f"Unable to locate testset job '{job_id}'",
                http_status=404,
            )

        logger.info(
            "testset.engine.start",
            extra={
                "context": {
                    "job_id": job.job_id,
                    "config_hash": job.config_hash,
                    "chunk_count": len(chunks),
                }
            },
        )
        self._repository.mark_running(job_id)

        try:
            samples, stats, generator_metadata = self._generator.generate(
                chunks=chunks,
                config=job.config,
            )
        except Exception as exc:  # pragma: no cover - generator errors propagated
            self._repository.mark_failed(
                job_id,
                error_code="generation_failed",
                error_message=str(exc),
            )
            raise

        if not samples:
            message = "Generator returned zero samples"
            self._repository.mark_failed(
                job_id,
                error_code="generation_empty",
                error_message=message,
            )
            raise ServiceError(
                error_code="generation_empty",
                message=message,
                http_status=422,
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        artifact_prefix = f"{self._storage_prefix}/{job.job_id}"

        sample_payload = self._encode_samples(samples)
        samples_key = f"{artifact_prefix}/samples.jsonl"
        samples_checksum = self._object_store.upload_bytes(
            self._bucket,
            samples_key,
            sample_payload,
            expected_checksum=compute_checksum(sample_payload),
        )

        metadata_document = self._build_metadata_document(
            job_id=job.job_id,
            config_hash=job.config_hash,
            method=job.method,
            stats=stats,
            generator_metadata=generator_metadata,
            sample_count=len(samples),
            checksum=samples_checksum,
            generated_at=timestamp,
        )
        metadata_payload = json.dumps(metadata_document, ensure_ascii=False, indent=2).encode("utf-8")
        metadata_key = f"{artifact_prefix}/metadata.json"
        self._object_store.upload_bytes(
            self._bucket,
            metadata_key,
            metadata_payload,
            expected_checksum=compute_checksum(metadata_payload),
        )

        artifact_paths = {
            "samples": samples_key,
            "metadata": metadata_key,
        }

        seed = metadata_document.get("seed")
        persona_count = int(metadata_document.get("persona_count") or 0)
        scenario_count = int(metadata_document.get("scenario_count") or 0)

        self._repository.mark_completed(
            job_id,
            sample_count=len(samples),
            persona_count=persona_count,
            scenario_count=scenario_count,
            seed=seed,
            artifact_prefix=artifact_prefix,
            artifact_paths=artifact_paths,
            metadata=metadata_document,
        )

        logger.info(
            "testset.engine.completed",
            extra={
                "context": {
                    "job_id": job.job_id,
                    "config_hash": job.config_hash,
                    "sample_count": len(samples),
                }
            },
        )

        return TestsetGenerationResult(
            sample_count=len(samples),
            artifact_paths=artifact_paths,
            stats=stats,
            metadata=metadata_document,
        )

    @staticmethod
    def _encode_samples(samples) -> bytes:
        records = [
            TestsetGenerationEngine._serialise_sample(sample) for sample in samples
        ]
        return "\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records
        ).encode("utf-8")

    @staticmethod
    def _serialise_sample(sample) -> Dict[str, Any]:
        payload = sample.eval_sample.model_dump(exclude_none=True)
        payload["synthesizer_name"] = sample.synthesizer_name
        return payload

    @staticmethod
    def _build_metadata_document(
        *,
        job_id: str,
        config_hash: str,
        method: str,
        stats: GenerationStats,
        generator_metadata: Mapping[str, Any] | Dict[str, Any],
        sample_count: int,
        checksum: str,
        generated_at: str,
    ) -> Dict[str, Any]:
        metadata = dict(generator_metadata) if isinstance(generator_metadata, Mapping) else {}
        persona = metadata.get("persona")
        scenarios = metadata.get("scenarios") or []
        persona_count = metadata.get("persona_count")
        if persona_count is None:
            persona_count = 1 if persona else 0
        scenario_count = metadata.get("scenario_count")
        if scenario_count is None:
            scenario_count = len(scenarios)
        strategies = metadata.get("strategies") or []

        return {
            "job_id": job_id,
            "config_hash": config_hash,
            "method": method,
            "sample_count": sample_count,
            "stats": asdict(stats),
            "seed": metadata.get("seed"),
            "persona": persona,
            "persona_count": persona_count,
            "scenarios": scenarios,
            "scenario_count": scenario_count,
            "strategies": strategies,
            "checksum": checksum,
            "generated_at": generated_at,
        }


__all__ = ["TestsetGenerationEngine", "TestsetGenerationResult"]
