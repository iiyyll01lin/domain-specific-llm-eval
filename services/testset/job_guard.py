from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from services.testset.repository import TestsetJob, TestsetRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobGuardResult:
    job: TestsetJob
    created: bool


class TestsetJobGuard:
    """Ensures testset job submissions are idempotent per configuration hash."""

    def __init__(self, repository: TestsetRepository) -> None:
        self._repository = repository

    def reserve(self, *, config_hash: str, config: Mapping[str, Any]) -> JobGuardResult:
        existing = self._repository.get_job_by_hash(config_hash)
        if existing and existing.status not in {"error", "cancelled"}:
            logger.debug(
                "testset.job_guard.duplicate",
                extra={
                    "context": {
                        "config_hash": config_hash,
                        "job_id": existing.job_id,
                        "status": existing.status,
                    }
                },
            )
            return JobGuardResult(job=existing, created=False)

        job = self._repository.create_job(config_hash=config_hash, config=dict(config))
        logger.debug(
            "testset.job_guard.created",
            extra={
                "context": {
                    "config_hash": config_hash,
                    "job_id": job.job_id,
                    "status": job.status,
                }
            },
        )
        return JobGuardResult(job=job, created=True)


__all__ = ["TestsetJobGuard", "JobGuardResult"]
