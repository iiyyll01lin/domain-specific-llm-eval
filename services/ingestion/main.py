from functools import lru_cache

from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError

from services.common.config import configure_service, settings
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.ingestion.repository import IngestionRepository
from services.ingestion.schemas import DocumentIngestionRequest, DocumentIngestionResponse

SERVICE_NAME = "ingestion-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@lru_cache
def get_repository() -> IngestionRepository:
    return IngestionRepository(settings.ingestion_db_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/readyz")
async def readyz():
    return {"status": "ready", "service": SERVICE_NAME}


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
    return {"service": SERVICE_NAME, "message": "ingestion service skeleton"}
