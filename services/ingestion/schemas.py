from pydantic import BaseModel, Field


class DocumentIngestionRequest(BaseModel):
    km_id: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)


class DocumentIngestionResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
