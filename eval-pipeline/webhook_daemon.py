from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

eval_pipeline_dir = Path(__file__).resolve().parent
if str(eval_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(eval_pipeline_dir))

from src.ui.dashboard_actions import build_pipeline_command

app = FastAPI(title="RAGAS Evaluation Webhook Daemon")


class WebhookPayload(BaseModel):
    source: str = Field(default="generic")
    event_type: str = Field(default="push")
    ref: str = Field(default="refs/heads/main")
    commit_sha: Optional[str] = None
    docs: int = Field(default=5, ge=1, le=500)
    samples: int = Field(default=50, ge=1, le=5000)
    config_path: Optional[str] = None


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
