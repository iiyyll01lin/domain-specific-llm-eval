from __future__ import annotations

from pydantic import BaseModel


class TestsetJobResponse(BaseModel):
    job_id: str
    config_hash: str
    status: str
    created_at: str
    duplicate: bool = False
    method: str


__all__ = ["TestsetJobResponse"]
