"""Insights Adapter service – normalization and KM summary export (TASK-040/045)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from services.common.config import configure_service
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware

logger = logging.getLogger(__name__)

SERVICE_NAME = "adapter-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class NormalizeSummaryRequest(BaseModel):
    run_id: str
    testset_id: str
    evaluation_item_count: int
    metrics_version: str
    kpis: Dict[str, Any] = {}
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None


class TestsetSummaryRequest(BaseModel):
    testset_id: str
    sample_count: int
    seed: int
    config_hash: str
    persona_count: int = 0
    scenario_count: int = 0
    duplicate: bool = False
    created_at: Optional[str] = None
    output_path: Optional[str] = None


class KgSummaryRequest(BaseModel):
    kg_id: str
    node_count: int
    relationship_count: int
    top_entities: Optional[List[Dict[str, Any]]] = None
    degree_histogram: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None
    output_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": SERVICE_NAME, "version": "0.1.0"}


@app.post("/normalize", status_code=status.HTTP_200_OK)
async def normalize_summary(req: NormalizeSummaryRequest) -> Dict[str, Any]:
    """
    Translate evaluation KPIs into a portal-friendly export_summary.json.

    If *output_path* is provided the summary is also written to disk.
    """
    from services.adapter.normalize import normalize_run_summary, write_export_summary

    summary = normalize_run_summary(
        run_id=req.run_id,
        testset_id=req.testset_id,
        kpis=req.kpis,
        evaluation_item_count=req.evaluation_item_count,
        metrics_version=req.metrics_version,
        created_at=req.created_at,
        completed_at=req.completed_at,
    )

    if req.output_path:
        try:
            write_export_summary(summary, req.output_path)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to write export summary: {exc}",
            ) from exc

    return summary


@app.post("/km-summaries/testset", status_code=status.HTTP_200_OK)
async def export_testset_summary(req: TestsetSummaryRequest) -> Dict[str, Any]:
    """Build and optionally persist a testset_summary_v0 document."""
    from services.adapter.km_export import build_testset_summary, write_summary

    summary = build_testset_summary(
        testset_id=req.testset_id,
        sample_count=req.sample_count,
        seed=req.seed,
        config_hash=req.config_hash,
        persona_count=req.persona_count,
        scenario_count=req.scenario_count,
        duplicate=req.duplicate,
        created_at=req.created_at,
    )

    if req.output_path:
        try:
            write_summary(summary, req.output_path)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to write testset summary: {exc}",
            ) from exc

    return summary


@app.post("/km-summaries/kg", status_code=status.HTTP_200_OK)
async def export_kg_summary(req: KgSummaryRequest) -> Dict[str, Any]:
    """Build and optionally persist a kg_summary_v0 document."""
    from services.adapter.km_export import build_kg_summary, write_summary

    summary = build_kg_summary(
        kg_id=req.kg_id,
        node_count=req.node_count,
        relationship_count=req.relationship_count,
        top_entities=req.top_entities,
        degree_histogram=req.degree_histogram,
        created_at=req.created_at,
    )

    if req.output_path:
        try:
            write_summary(summary, req.output_path)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to write KG summary: {exc}",
            ) from exc

    return summary
