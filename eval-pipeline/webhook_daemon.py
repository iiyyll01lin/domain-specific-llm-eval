from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    # The top-level outputs/ directory is mounted at /app/outputs in the
    # container (see docker-compose.services.yml webhook volume mapping).
    # Go up from eval-pipeline/ to the workspace root so kpis.json files
    # written by the pipeline runner are visible to the DriftStore.
    return str(eval_pipeline_dir.parent / "outputs")


def _auto_trigger_healing(drift_dict: Dict[str, Any]) -> None:
    """Fire off a background healing run in a daemon thread.

    Called automatically by the drift health-check job when the status
    transitions to DRIFTING — no HTTP request required.
    """
    import threading

    def _run() -> None:
        try:
            thread_id = str(uuid.uuid4())
            repo_root = str(eval_pipeline_dir.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            graph = _get_healing_graph()
            config = {"configurable": {"thread_id": thread_id}}
            initial_state: Dict[str, Any] = {
                "drift_result": drift_dict,
                "low_sc_queries": [],
                "knowledge_gaps": [],
                "retrieved_contexts": [],
                "proposed_patch": None,
                "proposal_id": None,
                "verification_report": None,
                "projected_sc_delta": None,
                "human_approved": None,
                "rejection_feedback": None,
                "iteration_count": 0,
                "max_iterations": 3,
                "status": "DIAGNOSING",
                "db_path": _KNOWLEDGE_GRAPH_DB,
                "_thread_id": thread_id,
                "abort_reason": None,
            }
            logger.info("Auto-healing triggered — thread=%s drift_status=%s", thread_id, drift_dict.get("status"))
            graph.invoke(initial_state, config=config)
        except Exception as exc:
            logger.error("Auto-healing background thread failed: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="agentic-healing")
    t.start()


def _drift_check_with_healing_hook(outputs_root: str) -> None:
    """Wrapper around run_check_now that fires the healing workflow on DRIFTING."""
    try:
        from services.eval.drift.scheduler import run_check_now, get_last_result
        run_check_now(outputs_root)
        result = get_last_result()
        if result and result.status == "DRIFTING":
            logger.info("DRIFTING detected — spawning agentic healing workflow.")
            _auto_trigger_healing(result.to_dict())
    except Exception as exc:
        logger.error("Drift check with healing hook failed: %s", exc)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Start the drift-detection scheduler on startup; stop it on shutdown."""
    outputs_root = _drift_outputs_root()
    scheduler = None
    try:
        from services.eval.drift.scheduler import create_scheduler

        # Run an immediate check (with healing hook) so /drift-status can
        # respond before the first scheduled tick.
        _drift_check_with_healing_hook(outputs_root)

        scheduler = create_scheduler(outputs_root)
        # Override the default drift_check job with our hook-enhanced version
        scheduler.add_job(
            _drift_check_with_healing_hook,
            trigger="interval",
            hours=6,
            args=[outputs_root],
            id="drift_check_healing",
            name="GCR Drift Check + Healing Hook",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Drift scheduler started (with healing hook).")

        # Prune stale proposals on startup
        try:
            _get_proposal_store().prune_expired()
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Agentic Self-Healing routes
# ---------------------------------------------------------------------------

# Module-level singletons — initialised lazily on first request so the module
# can be imported in tests without requiring langgraph / services packages.
_healing_graph: Any = None
_proposal_store: Any = None

_HEALING_CHECKPOINTER_DB = str(eval_pipeline_dir.parent / "outputs" / "healing" / "checkpoints.db")
_HEALING_PROPOSALS_DB = str(eval_pipeline_dir.parent / "outputs" / "healing" / "proposals.db")
_KNOWLEDGE_GRAPH_DB = str(eval_pipeline_dir.parent / "outputs" / "knowledge_graph.db")


def _get_healing_graph() -> Any:
    """Return the cached LangGraph healing graph, building it on first call."""
    global _healing_graph
    if _healing_graph is None:
        try:
            # Ensure the workspace root is on sys.path so services.* is importable.
            repo_root = str(eval_pipeline_dir.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from services.eval.agentic_healing.graph import build_healing_graph
            _healing_graph = build_healing_graph(_HEALING_CHECKPOINTER_DB)
        except Exception as exc:
            logger.error("Failed to build healing graph: %s", exc)
            raise RuntimeError(f"Agentic healing graph unavailable: {exc}") from exc
    return _healing_graph


def _get_proposal_store() -> Any:
    """Return the cached ProposalStore, creating it on first call."""
    global _proposal_store
    if _proposal_store is None:
        repo_root = str(eval_pipeline_dir.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from services.eval.agentic_healing.staging import ProposalStore
        _proposal_store = ProposalStore(_HEALING_PROPOSALS_DB)
    return _proposal_store


class HealingTriggerPayload(BaseModel):
    """Request body for manually triggering an agentic healing run."""

    low_sc_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of LowScoreQuery dicts from recent evaluation runs.",
    )
    db_path: Optional[str] = Field(
        default=None,
        description="Path to the SQLiteGraphStore .db file. Defaults to server-side config.",
    )
    max_iterations: int = Field(default=3, ge=1, le=10)


class ApprovalPayload(BaseModel):
    approved: bool = Field(description="True to approve the patch; False to reject it.")
    rejection_feedback: Optional[str] = Field(
        default=None,
        description="Optional reason for rejection, forwarded to the Diagnostician on next cycle.",
    )


def _run_healing_background(initial_state: Dict[str, Any], thread_id: str) -> None:
    """Background task: run the healing graph until the commit interrupt."""
    try:
        graph = _get_healing_graph()
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state, config=config)
        logger.info(
            "Healing graph paused — thread=%s status=%s proposal=%s",
            thread_id,
            (result or {}).get("status", "?"),
            (result or {}).get("proposal_id", "—"),
        )

        # Mirror the final waiting state into ProposalStore so the frontend
        # can read it without accessing the LangGraph checkpointer directly.
        if result and result.get("proposal_id") and result.get("status") in (
            "AWAITING_APPROVAL", "VERIFYING", None, ""
        ):
            patch = result.get("proposed_patch") or {}
            proposal_id = result["proposal_id"]
            try:
                store = _get_proposal_store()
                # Update the proposal's state_json with the final waiting state.
                store.write_proposal(
                    proposal_id=proposal_id,
                    thread_id=thread_id,
                    state=result,
                )
            except Exception as exc:
                logger.warning("Could not update ProposalStore after graph pause: %s", exc)

    except Exception as exc:
        logger.error("Healing background task failed for thread=%s: %s", thread_id, exc)


@app.post("/api/v1/healing/trigger")
async def trigger_healing(
    payload: HealingTriggerPayload,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Manually trigger an agentic Knowledge Graph healing run.

    The run executes entirely in a background task.  It will pause automatically
    before the ``commit`` node and wait for human approval via
    ``POST /api/v1/healing/proposals/{proposal_id}/approve``.

    Also triggered automatically by the drift scheduler when status = DRIFTING.
    """
    thread_id = str(uuid.uuid4())
    db_path = payload.db_path or _KNOWLEDGE_GRAPH_DB

    # Attempt to include current drift context
    drift_result: Dict[str, Any] = {"status": "MANUAL_TRIGGER", "metrics": {}}
    try:
        from services.eval.drift.scheduler import get_last_result
        last = get_last_result()
        if last:
            drift_result = last.to_dict()
    except Exception:
        pass

    initial_state: Dict[str, Any] = {
        "drift_result": drift_result,
        "low_sc_queries": payload.low_sc_queries,
        "knowledge_gaps": [],
        "retrieved_contexts": [],
        "proposed_patch": None,
        "proposal_id": None,
        "verification_report": None,
        "projected_sc_delta": None,
        "human_approved": None,
        "rejection_feedback": None,
        "iteration_count": 0,
        "max_iterations": payload.max_iterations,
        "status": "DIAGNOSING",
        "db_path": db_path,
        "_thread_id": thread_id,
        "abort_reason": None,
    }

    background_tasks.add_task(_run_healing_background, initial_state, thread_id)
    logger.info("Healing run queued — thread=%s db=%s", thread_id, db_path)

    return {
        "message": "Agentic healing workflow started.",
        "thread_id": thread_id,
        "status": "QUEUED",
    }


@app.get("/api/v1/healing/proposals")
async def list_healing_proposals() -> Dict[str, Any]:
    """Return all pending (non-expired) repair proposals awaiting human approval."""
    try:
        store = _get_proposal_store()
        store.prune_expired()
        proposals = store.list_pending()
    except Exception as exc:
        logger.error("Could not list proposals: %s", exc)
        raise HTTPException(status_code=503, detail=f"ProposalStore unavailable: {exc}") from exc

    # Flatten: expose only the patch data + metadata (not the full state blob)
    items = []
    for p in proposals:
        state = p.get("state") or {}
        patch = state.get("proposed_patch") or {}
        items.append(
            {
                "proposal_id": p["proposal_id"],
                "thread_id": p["thread_id"],
                "status": p["status"],
                "created_at": p["created_at"],
                "expires_at": p["expires_at"],
                "proposed_nodes_count": len(patch.get("proposed_nodes", [])),
                "proposed_edges_count": len(patch.get("proposed_edges", [])),
                "estimated_sc_delta": patch.get("estimated_sc_delta", 0.0),
                "rationale": patch.get("rationale", ""),
                "proposed_nodes": patch.get("proposed_nodes", []),
                "proposed_edges": patch.get("proposed_edges", []),
                "knowledge_gaps": state.get("knowledge_gaps", []),
                "verification_report": state.get("verification_report"),
                "drift_status": (state.get("drift_result") or {}).get("status"),
            }
        )

    return {"proposals": items, "count": len(items)}


@app.post("/api/v1/healing/proposals/{proposal_id}/approve")
async def approve_healing_proposal(
    proposal_id: str,
    payload: ApprovalPayload,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Approve or reject a pending repair proposal.

    Approval resumes the paused LangGraph thread, allowing the ``commit`` node
    to execute and permanently upsert the proposed nodes/edges.
    Rejection re-routes to the Diagnostician with optional feedback.
    """
    store = _get_proposal_store()
    proposal = store.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal '{proposal_id}' not found.")
    if proposal["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Proposal is in state '{proposal['status']}', not 'pending'.",
        )

    thread_id = proposal["thread_id"]
    state = proposal.get("state") or {}

    if not payload.approved:
        store.update_status(proposal_id, "rejected")
        logger.info("Proposal %s rejected by user.", proposal_id)
        return {"message": "Proposal rejected.", "proposal_id": proposal_id, "status": "rejected"}

    # --- Approve: resume LangGraph checkpoint ---
    def _resume_and_commit() -> None:
        try:
            graph = _get_healing_graph()
            config = {"configurable": {"thread_id": thread_id}}
            # Inject approval into the checkpoint state
            graph.update_state(
                config,
                {
                    "human_approved": True,
                    "rejection_feedback": None,
                    "status": "AWAITING_APPROVAL",
                },
            )
            result = graph.invoke(None, config=config)
            final_status = (result or {}).get("status", "UNKNOWN")
            logger.info(
                "Healing commit complete — thread=%s proposal=%s final_status=%s",
                thread_id, proposal_id, final_status,
            )
            store.update_status(proposal_id, "committed" if final_status == "COMMITTED" else "approved")
        except Exception as exc:
            logger.error("Resume after approval failed for proposal=%s: %s", proposal_id, exc)
            store.update_status(proposal_id, "error")

    background_tasks.add_task(_resume_and_commit)
    logger.info("Proposal %s approved — resuming graph thread=%s.", proposal_id, thread_id)

    return {
        "message": "Proposal approved. Graph repair committing in the background.",
        "proposal_id": proposal_id,
        "thread_id": thread_id,
        "status": "COMMITTING",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
