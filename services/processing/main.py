from functools import lru_cache

from fastapi import Depends, FastAPI, Response, status
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from services.common.config import configure_service, settings
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.ingestion.repository import IngestionRepository
from services.processing.repository import ProcessingRepository
from services.processing.schemas import ProcessingJobRequest, ProcessingJobResponse

SERVICE_NAME = "processing-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@lru_cache
def get_repository() -> ProcessingRepository:
    return ProcessingRepository(settings.processing_db_path)


@lru_cache
def get_document_repository() -> IngestionRepository:
    return IngestionRepository(settings.ingestion_db_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/metrics")
async def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/process-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessingJobResponse,
)
async def submit_processing_job(
    payload: ProcessingJobRequest,
    repository: ProcessingRepository = Depends(get_repository),
    document_repository: IngestionRepository = Depends(get_document_repository),
):
    document = document_repository.get_document(payload.document_id)
    if document is None:
        raise ServiceError(
            error_code="document_not_found",
            message=f"Document {payload.document_id} not found",
            http_status=status.HTTP_404_NOT_FOUND,
        )
    job = repository.create_job(document_id=document.document_id, profile_hash=payload.profile_hash)
    return ProcessingJobResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        profile_hash=job.profile_hash,
        status=job.status,
        created_at=job.created_at,
    )


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "message": "processing service skeleton"}
