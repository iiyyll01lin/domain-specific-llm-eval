from pydantic import BaseModel, Field


class ProcessingJobRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=64)
    profile_hash: str = Field(..., min_length=1, max_length=128)


class ProcessingJobResponse(BaseModel):
    job_id: str
    document_id: str
    profile_hash: str
    status: str
    created_at: str
