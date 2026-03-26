from __future__ import annotations

from pydantic import BaseModel


class EvaluationRunResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    duplicate: bool = False


__all__ = ["EvaluationRunResponse"]
