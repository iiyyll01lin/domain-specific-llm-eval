from functools import lru_cache

from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError

from services.common.config import settings
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.ingestion.repository import IngestionRepository
from services.ingestion.schemas import DocumentIngestionRequest, DocumentIngestionResponse

app = FastAPI(title="ingestion-service")
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@lru_cache
def get_repository() -> IngestionRepository:
    return IngestionRepository(settings.ingestion_db_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.service_name}


@app.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentIngestionResponse,
)
async def submit_document(
    payload: DocumentIngestionRequest,
    repository: IngestionRepository = Depends(get_repository),
):
    job = repository.create_job(km_id=payload.km_id, version=payload.version)
    return DocumentIngestionResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
    )


@app.get("/")
async def root():
    return {"service": "ingestion", "message": "ingestion service skeleton"}
