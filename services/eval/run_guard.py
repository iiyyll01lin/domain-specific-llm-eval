from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.eval.repository import EvaluationRun, EvaluationRunRepository


@dataclass(frozen=True)
class RunGuardResult:
    run: EvaluationRun
    created: bool


class EvaluationRunGuard:
    """Ensures evaluation run submissions are idempotent for active runs."""

    def __init__(self, repository: EvaluationRunRepository) -> None:
        self._repository = repository

    def reserve(
        self,
        *,
        testset_id: str,
        profile: str,
    rag_endpoint: Optional[str],
        timeout_seconds: int,
        max_retries: int,
    ) -> RunGuardResult:
        existing = self._repository.get_active_run(testset_id=testset_id, profile=profile)
        if existing is not None:
            return RunGuardResult(run=existing, created=False)

        run = self._repository.create_run(
            testset_id=testset_id,
            profile=profile,
            rag_endpoint=rag_endpoint,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return RunGuardResult(run=run, created=True)


__all__ = ["EvaluationRunGuard", "RunGuardResult"]
