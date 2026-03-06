"""Reporting service – HTML/PDF report generation and listing (TASK-041/042)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
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

SERVICE_NAME = "reporting-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_reports_dir() -> Path:
    from services.common.config import settings
    return Path(getattr(settings, "reports_dir", "reports"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    run_id: str
    testset_id: str
    metrics_version: str = "0.0.0"
    evaluation_item_count: int = 0
    metrics: List[Dict[str, Any]] = []
    counts: Dict[str, Any] = {}
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    template: str = "executive"
    generate_pdf: bool = False


class ReportItem(BaseModel):
    run_id: str
    template: str
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None
    html_available: bool = False
    pdf_available: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/readyz")
async def readyz() -> Dict[str, str]:
    return {"status": "ready", "service": SERVICE_NAME}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": SERVICE_NAME, "version": "0.1.0"}


@app.post("/reports", status_code=status.HTTP_202_ACCEPTED)
async def generate_report(
    req: GenerateReportRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Trigger report generation for a completed evaluation run."""
    summary = {
        "run_id": req.run_id,
        "testset_id": req.testset_id,
        "metrics_version": req.metrics_version,
        "evaluation_item_count": req.evaluation_item_count,
        "metrics": req.metrics,
        "counts": req.counts,
        "created_at": req.created_at,
        "completed_at": req.completed_at,
    }
    output_dir = _get_reports_dir() / req.run_id

    background_tasks.add_task(
        _run_generate_report,
        summary=summary,
        output_dir=str(output_dir),
        template=req.template,
        generate_pdf=req.generate_pdf,
    )
    return {"run_id": req.run_id, "status": "generating", "output_dir": str(output_dir)}


@app.get("/reports", response_model=List[ReportItem])
async def list_reports() -> List[ReportItem]:
    """List all available HTML/PDF reports."""
    reports_dir = _get_reports_dir()
    items: List[ReportItem] = []
    if not reports_dir.exists():
        return items

    for run_dir in sorted(reports_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for template in ("executive", "technical"):
            html_p = run_dir / f"{run_id}_{template}.html"
            pdf_p = run_dir / f"{run_id}_{template}.pdf"
            if html_p.exists() or pdf_p.exists():
                items.append(
                    ReportItem(
                        run_id=run_id,
                        template=template,
                        html_path=str(html_p) if html_p.exists() else None,
                        pdf_path=str(pdf_p) if pdf_p.exists() else None,
                        html_available=html_p.exists(),
                        pdf_available=pdf_p.exists(),
                    )
                )
    return items


@app.get("/reports/{run_id}/{template}/html")
async def get_html_report(run_id: str, template: str) -> FileResponse:
    """Download the HTML report for a run."""
    _validate_path_segment(run_id)
    _validate_path_segment(template)
    path = _get_reports_dir() / run_id / f"{run_id}_{template}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/reports/{run_id}/{template}/pdf")
async def get_pdf_report(run_id: str, template: str) -> FileResponse:
    """Download the PDF report for a run; falls back to HTML 404 hint."""
    _validate_path_segment(run_id)
    _validate_path_segment(template)
    path = _get_reports_dir() / run_id / f"{run_id}_{template}.pdf"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not available. Use /html endpoint for the HTML version.",
        )
    return FileResponse(str(path), media_type="application/pdf")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_generate_report(
    *,
    summary: Dict[str, Any],
    output_dir: str,
    template: str,
    generate_pdf: bool,
) -> None:
    try:
        from services.reporting.pdf import generate_report
        paths = generate_report(
            summary,
            output_dir,
            template=template,  # type: ignore[arg-type]
            generate_pdf=generate_pdf,
        )
        logger.info("report generated", extra={"paths": paths, "run_id": summary.get("run_id")})
    except Exception as exc:
        logger.error("report generation failed: %s", exc, extra={"run_id": summary.get("run_id")})


def _validate_path_segment(value: str) -> None:
    """Prevent path traversal by validating segment is safe."""
    if not value or "/" in value or "\\" in value or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid path segment: {value!r}")
