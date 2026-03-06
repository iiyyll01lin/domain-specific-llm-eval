import logging
from functools import lru_cache
from typing import Dict

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pathlib import Path

from services.common.config import configure_service
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.eval.metrics import (
    EvaluationRunMetricsRecorder,
    PrometheusEvaluationRunMetrics,
)
from services.eval.metrics.loader import MetricRegistry, load_metric_registry
from services.eval.repository import EvaluationRunRepository
from services.eval.persistence_pipeline import EvaluationPersistencePipeline
from services.eval.run_guard import EvaluationRunGuard, RunGuardResult
from services.eval.schemas import EvaluationRunResponse
from services.eval.validation import (
    EvaluationRunCreateRequest,
    EvaluationRunValidator,
    get_default_profiles,
)
from services.testset.repository import TestsetRepository

logger = logging.getLogger(__name__)

SERVICE_NAME = "eval-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@lru_cache
def get_repository() -> EvaluationRunRepository:
    from services.common.config import settings

    return EvaluationRunRepository(settings.eval_db_path)


@lru_cache
def get_testset_repository() -> TestsetRepository:
    from services.common.config import settings

    return TestsetRepository(settings.testset_db_path)


async def _testset_exists(testset_id: str) -> bool:
    repository = get_testset_repository()
    job = repository.get_job(testset_id)
    return job is not None


@lru_cache
def get_validator() -> EvaluationRunValidator:
    return EvaluationRunValidator(
        testset_exists_fn=_testset_exists,
        available_profiles=get_default_profiles(),
    )


@lru_cache
def get_guard() -> EvaluationRunGuard:
    return EvaluationRunGuard(get_repository())


@lru_cache
def get_metric_registry() -> MetricRegistry:
    return load_metric_registry()


@lru_cache
def get_metrics() -> EvaluationRunMetricsRecorder:
    return PrometheusEvaluationRunMetrics()


@lru_cache
def get_output_root() -> Path:
    from services.common.config import settings

    return Path(settings.eval_outputs_dir)

@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/readyz")
async def readyz():
    return {"status": "ready", "service": SERVICE_NAME}


@app.get("/metrics")
async def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/eval-runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=EvaluationRunResponse,
)
async def submit_eval_run(
    payload: EvaluationRunCreateRequest,
    request: Request,
    guard: EvaluationRunGuard = Depends(get_guard),
    validator: EvaluationRunValidator = Depends(get_validator),
    metrics_recorder: EvaluationRunMetricsRecorder = Depends(get_metrics),
) -> EvaluationRunResponse:
    await validator.validate(payload)

    result: RunGuardResult = guard.reserve(
        testset_id=payload.testset_id,
        profile=payload.profile,
        rag_endpoint=payload.rag_endpoint,
        timeout_seconds=payload.timeout_seconds or 30,
        max_retries=payload.max_retries or 3,
    )

    outcome = "created" if result.created else "duplicate"
    metrics_recorder.record(outcome)

    trace_id = getattr(request.state, "trace_id", None)
    logger.info(
        "eval.run_submitted",
        extra={
            "trace_id": trace_id,
            "context": {
                "run_id": result.run.run_id,
                "testset_id": payload.testset_id,
                "profile": payload.profile,
                "result": outcome,
            },
        },
    )

    return EvaluationRunResponse(
        run_id=result.run.run_id,
        status=result.run.status,
        created_at=result.run.created_at,
        duplicate=not result.created,
    )


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "message": "evaluation service skeleton"}

def build_persistence_pipeline(run_id: str) -> EvaluationPersistencePipeline:
    output_root = get_output_root()
    return EvaluationPersistencePipeline(run_id, output_root / run_id)
