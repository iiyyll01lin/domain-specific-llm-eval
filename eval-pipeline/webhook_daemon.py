from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

eval_pipeline_dir = Path(__file__).resolve().parent
if str(eval_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(eval_pipeline_dir))

from src.ui.dashboard_actions import build_pipeline_command
from src.ui.reviewer_actions import _build_service


# ---------------------------------------------------------------------------
# Drift scheduler — started lazily so the module can be imported in tests
# without spawning background threads.
# ---------------------------------------------------------------------------

def _drift_outputs_root() -> str:
    return str(eval_pipeline_dir / "outputs")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Start the drift-detection scheduler on startup; stop it on shutdown."""
    outputs_root = _drift_outputs_root()
    scheduler = None
    try:
        from services.eval.drift.scheduler import create_scheduler, run_check_now

        # Run an immediate check so /drift-status can respond before the first
        # scheduled tick.
        run_check_now(outputs_root)

        scheduler = create_scheduler(outputs_root)
        scheduler.start()
        logger.info("Drift scheduler started.")
    except ImportError:
        logger.warning(
            "services.eval.drift not importable — drift monitoring disabled. "
            "Ensure the repo root is on PYTHONPATH and apscheduler is installed."
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to start drift scheduler: %s", exc)

    yield

    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
            logger.info("Drift scheduler stopped.")
        except Exception as exc:  # pragma: no cover
            logger.error("Error stopping drift scheduler: %s", exc)


app = FastAPI(title="RAGAS Evaluation Webhook Daemon", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class WebhookPayload(BaseModel):
    source: str = Field(default="generic")
    event_type: str = Field(default="push")
    ref: str = Field(default="refs/heads/main")
    commit_sha: Optional[str] = None
    docs: int = Field(default=5, ge=1, le=500)
    samples: int = Field(default=50, ge=1, le=5000)
    config_path: Optional[str] = None


class ReviewerDecisionPayload(BaseModel):
    review_id: Optional[str] = None
    index: Optional[int] = None
    question: str = Field(default="")
    approved: bool = Field(default=True)
    score: Optional[float] = None
    notes: str = Field(default="")
    reviewer: str = Field(default="reviewer")
    resolution: str = Field(default="resolved")


def _webhook_log_path() -> Path:
    outputs_dir = eval_pipeline_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / "webhook_events.jsonl"


def log_webhook_event(payload: WebhookPayload, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    event = {
        "source": payload.source,
        "event_type": payload.event_type,
        "ref": payload.ref,
        "commit_sha": payload.commit_sha,
        "docs": payload.docs,
        "samples": payload.samples,
        "status": status,
        "details": details or {},
    }
    with open(_webhook_log_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _review_base_dir() -> Path:
    return eval_pipeline_dir


def _reviewer_auth_context(
    reviewer_token: Optional[str],
    reviewer_id: Optional[str],
    tenant_id: Optional[str],
):
    service = _build_service(_review_base_dir())
    try:
        auth = service.authenticate(
            token=str(reviewer_token or ""),
            reviewer_id=reviewer_id,
            tenant_id=tenant_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return service, auth


def should_trigger_evaluation(payload: WebhookPayload) -> bool:
    return payload.event_type in {"push", "manual"} and payload.ref.startswith("refs/heads/")


def run_evaluation_pipeline(payload: WebhookPayload) -> subprocess.CompletedProcess[str]:
    """Trigger the pipeline asynchronously on incoming git pushes / hooks"""
    command = build_pipeline_command(
        docs=payload.docs,
        samples=payload.samples,
        config_path=payload.config_path,
    )
    log_webhook_event(payload, "started", {"command": command})
    result = subprocess.run(
        command,
        cwd=str(eval_pipeline_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    log_webhook_event(
        payload,
        "completed" if result.returncode == 0 else "failed",
        {"returncode": result.returncode, "stderr": result.stderr[-2000:]},
    )
    return result


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def trigger_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Medium/Low Priority: Run pipeline on git pushes or other webhooks.
    """
    if not should_trigger_evaluation(payload):
        log_webhook_event(payload, "ignored")
        return {"message": "Webhook ignored.", "status": "ignored"}

    background_tasks.add_task(run_evaluation_pipeline, payload)
    log_webhook_event(payload, "queued")
    return {
        "message": "Evaluation pipeline triggered.",
        "status": "queued",
        "docs": payload.docs,
        "samples": payload.samples,
    }


@app.get("/reviews")
async def list_reviews(
    status: str = "pending",
    include_resolved: bool = False,
    x_reviewer_token: Optional[str] = Header(default=None),
    x_reviewer_id: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    service, auth = _reviewer_auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id)
    return service.list_reviews(
        auth,
        status=status if status else None,
        include_resolved=include_resolved,
    )


@app.get("/reviews/summary")
async def review_summary(
    x_reviewer_token: Optional[str] = Header(default=None),
    x_reviewer_id: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    service, auth = _reviewer_auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id)
    return service.get_summary(auth)


@app.post("/reviews/submit")
async def submit_review(
    payload: ReviewerDecisionPayload,
    x_reviewer_token: Optional[str] = Header(default=None),
    x_reviewer_id: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    service, auth = _reviewer_auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id)
    return service.submit_review(auth, payload.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Drift status endpoint
# ---------------------------------------------------------------------------

@app.get("/api/v1/drift-status")
async def drift_status() -> Dict[str, Any]:
    """Return the most recent GCR data-drift analysis result.

    The result is produced by the background ``DriftDetector`` scheduler and
    cached in-process.  A cold response (before the first check completes)
    returns ``{"status": "PENDING"}``.
    """
    try:
        from services.eval.drift.scheduler import get_last_result

        result = get_last_result()
        if result is None:
            return {"status": "PENDING", "message": "Drift check not yet completed."}
        return result.to_dict()
    except ImportError:
        return {
            "status": "UNAVAILABLE",
            "message": "Drift monitoring module not installed.",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
