"""Reporting service – HTML/PDF report generation and listing (TASK-041/042)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
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


# ---------------------------------------------------------------------------
# Auto-Insights: LLM-powered executive summary (TASK-060)
# ---------------------------------------------------------------------------

_INSIGHTS_SYSTEM_PROMPT = """\
You are an expert RAG (Retrieval-Augmented Generation) system evaluator. \
Your role is to produce a concise, actionable "System Health Report" in Markdown \
for an engineering or product team reviewing an evaluation run.

## Metric Reference Guide

### Graph Context Relevance (GCR) Suite
These metrics assess how well the Knowledge Graph supports retrieval quality.

| Metric | Key | Direction | Interpretation |
|---|---|---|---|
| GCR Composite | `gcr_score` | Higher is better (0–1) | Weighted average of Se, Sc, Ph. A score above 0.7 is healthy. Below 0.5 indicates structural KG problems. |
| Entity Overlap | `entity_overlap` (Sₑ) | Higher is better (0–1) | Measures named-entity intersection between retrieved contexts and reference answers. Low Sₑ (< 0.4) means the graph is surfacing documents that do not contain the relevant facts — likely a chunking or embedding issue. |
| Structural Connectivity | `structural_connectivity` (Sᶜ) | Higher is better (0–1) | Measures how well-connected the retrieved nodes are in the knowledge graph. High Sᶜ (> 0.7) means excellent multi-hop retrieval paths exist. Low Sᶜ means the graph is fragmented — add more relationship builders or lower similarity thresholds. |
| Hub Noise Penalty | `hub_noise_penalty` (Pₕ) | **Lower is better (0–1)** | Penalizes retrieval paths that pass through generic "hub" nodes (high-degree, low-information nodes like "the system", "data", "process"). A high Pₕ (> 0.3) means the graph is cluttered with irrelevant generic nodes. Action: prune hub nodes, raise minimum edge similarity thresholds, or apply domain stop-node filters. |

### Standard RAGAS Metrics

| Metric | Direction | Interpretation |
|---|---|---|
| `Faithfulness` | Higher is better (0–1) | Fraction of answer claims verifiable in retrieved contexts. Low faithfulness (< 0.7) indicates hallucination. |
| `AnswerRelevancy` | Higher is better (0–1) | Semantic alignment between question and generated answer. Low score may reflect off-topic generation or poor retrieval. |
| `ContextPrecision` | Higher is better (0–1) | Fraction of retrieved chunks that are relevant. Low precision means too many irrelevant chunks are being retrieved. |
| `ContextRecall` | Higher is better (0–1) | Fraction of reference information covered by retrieved chunks. Low recall means the retriever is missing key documents. |
| `ContextualKeywordMean` | Higher is better (0–1) | Domain-keyword coverage in retrieved contexts. Low score suggests domain-specific vocabulary is underrepresented in the corpus. |

## Output Format Requirements
- Begin with a 1-sentence overall health verdict (e.g., "🟡 **At Risk**: ...").
- Use the emoji: 🟢 (Healthy), 🟡 (At Risk), 🔴 (Critical).
- Write 3–5 concise bullet points under "**Key Findings**", citing specific metric values.
- Write 3–5 prioritized items under "**Recommended Actions**", ordered by impact.
- Keep the total response under 400 words.
- Do NOT include raw JSON or repeat every single metric value — focus only on the \
most diagnostically significant values and patterns.
"""


class InsightsRequest(BaseModel):
    run_id: Optional[str] = None
    kpis: Dict[str, float]
    counts: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    failing_metrics: Optional[List[str]] = None
    thresholds: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # override; defaults to env INSIGHTS_LLM_MODEL


class InsightsResponse(BaseModel):
    run_id: Optional[str]
    summary: str
    model_used: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def _build_user_message(req: InsightsRequest) -> str:
    """Serialise the request payload into a structured user prompt."""
    lines: List[str] = [
        f"**Run ID:** {req.run_id or 'unknown'}",
        f"**Overall Verdict:** {req.verdict or 'N/A'}",
    ]
    if req.failing_metrics:
        lines.append(f"**Failing Metrics:** {', '.join(req.failing_metrics)}")
    if req.counts:
        lines.append(f"**Item Count:** {req.counts}")

    lines.append("\n### KPI Values\n")
    for key, val in sorted(req.kpis.items()):
        direction_hint = "(lower is better)" if key == "hub_noise_penalty" else "(higher is better)"
        threshold_info = ""
        if req.thresholds and key in req.thresholds:
            th = req.thresholds[key]
            w = th.get("warning", "?")
            c = th.get("critical", "?")
            threshold_info = f" | threshold: warn={w}, crit={c}"
        lines.append(f"- `{key}`: {val:.4f} {direction_hint}{threshold_info}")

    return "\n".join(lines)


@app.post("/api/v1/insights/generate", response_model=InsightsResponse)
async def generate_insights_summary(req: InsightsRequest) -> InsightsResponse:
    """
    Call an LLM to produce a human-readable System Health Report from aggregated
    evaluation KPIs.  Requires OPENAI_API_KEY (or INSIGHTS_API_KEY) to be set.
    Supports model override via INSIGHTS_LLM_MODEL env var (default: gpt-4o-mini).
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("INSIGHTS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LLM API key not configured. "
                "Set OPENAI_API_KEY (or INSIGHTS_API_KEY) in the environment to enable AI Insights."
            ),
        )

    try:
        from openai import AsyncOpenAI, APIError, AuthenticationError
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="openai package is not installed in this environment.",
        )

    model = req.model or os.getenv("INSIGHTS_LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("INSIGHTS_API_BASE_URL") or None  # None → default OpenAI endpoint

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    user_message = _build_user_message(req)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _INSIGHTS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=600,
        )
    except AuthenticationError as exc:
        logger.error("LLM authentication failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM authentication failed — check that OPENAI_API_KEY is valid.",
        )
    except APIError as exc:
        logger.error("LLM API error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API returned an error: {exc.message}",
        )
    except Exception as exc:
        logger.exception("Unexpected error calling LLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unexpected error communicating with the LLM service.",
        )

    choice = response.choices[0]
    summary_text = choice.message.content or ""
    usage = response.usage

    logger.info(
        "insights generated",
        extra={
            "run_id": req.run_id,
            "model": model,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        },
    )

    return InsightsResponse(
        run_id=req.run_id,
        summary=summary_text,
        model_used=model,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )

