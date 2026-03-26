import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import Body, Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from services.common.config import configure_service
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.testset.job_guard import JobGuardResult, TestsetJobGuard
from services.testset.metrics import PrometheusTestsetJobMetrics, TestsetJobMetricsRecorder
from services.testset.repository import TestsetRepository
from services.testset.schemas import TestsetJobResponse
from services.testset.validation import validate_testset_config
from services.testset.config_normalizer import compute_config_hash

logger = logging.getLogger(__name__)

SERVICE_NAME = "testset-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@lru_cache
def get_repository() -> TestsetRepository:
    from services.common.config import settings

    return TestsetRepository(settings.testset_db_path)


@lru_cache
def get_job_guard() -> TestsetJobGuard:
    return TestsetJobGuard(get_repository())


@lru_cache
def get_metrics() -> TestsetJobMetricsRecorder:
    return PrometheusTestsetJobMetrics()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/readyz")
async def readyz() -> Dict[str, str]:
    return {"status": "ready", "service": SERVICE_NAME}


@app.get("/metrics")
async def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/testset-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TestsetJobResponse,
)
async def submit_testset_job(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    guard: TestsetJobGuard = Depends(get_job_guard),
    metrics_recorder: TestsetJobMetricsRecorder = Depends(get_metrics),
) -> TestsetJobResponse:
    sanitized = validate_testset_config(payload)
    config_hash = compute_config_hash(sanitized)
    result: JobGuardResult = guard.reserve(config_hash=config_hash, config=sanitized)

    outcome = "created" if result.created else "duplicate"
    metrics_recorder.record(outcome)

    trace_id = getattr(request.state, "trace_id", None)
    logger.info(
        "testset.job_submitted",
        extra={
            "trace_id": trace_id,
            "context": {
                "job_id": result.job.job_id,
                "config_hash": config_hash,
                "method": result.job.method,
                "result": outcome,
            },
        },
    )

    return TestsetJobResponse(
        job_id=result.job.job_id,
        config_hash=config_hash,
        status=result.job.status,
        created_at=result.job.created_at,
        duplicate=not result.created,
        method=result.job.method,
    )


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "message": "testset service skeleton"}
