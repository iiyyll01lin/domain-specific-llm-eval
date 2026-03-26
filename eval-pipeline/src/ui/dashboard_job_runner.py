from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class DashboardJobRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.jobs_dir = self.base_dir / "outputs" / "dashboard_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self, docs: int, samples: int, config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        created_at = datetime.utcnow().isoformat() + "Z"
        run_dir_name = f"pure_ragas_run_{job_id}"
        stdout_path = self.jobs_dir / f"{job_id}.stdout.log"
        stderr_path = self.jobs_dir / f"{job_id}.stderr.log"
        command = [
            sys.executable,
            "run_pure_ragas_pipeline.py",
            "--docs",
            str(int(docs)),
            "--samples",
            str(int(samples)),
        ]
        if config_path:
            command.extend(["--config", str(Path(config_path))])

        status = {
            "job_id": job_id,
            "status": "queued",
            "docs": int(docs),
            "samples": int(samples),
            "config_path": config_path,
            "command": command,
            "created_at": created_at,
            "updated_at": created_at,
            "run_dir": str(self.base_dir / "outputs" / run_dir_name),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        self._write_status(job_id, status)

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(self.base_dir),
                stdout=stdout_handle,
                stderr=stderr_handle,
                env={
                    **os.environ,
                    "PIPELINE_RUN_ID": job_id,
                },
            )
        status.update({"status": "running", "pid": process.pid})
        self._write_status(job_id, status)
        return status

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        status = self._read_status(job_id)
        pid = status.get("pid")
        if status.get("status") != "running" or not isinstance(pid, int):
            return self._augment_status(status)

        proc_path = Path("/proc") / str(pid)
        if proc_path.exists():
            return self._augment_status(status)

        updated_status = {**status, "status": self._resolve_completed_state(status)}
        self._write_status(job_id, updated_status)
        return self._augment_status(updated_status)

    def list_jobs(self) -> Dict[str, Any]:
        jobs = []
        for status_file in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            jobs.append(self.get_job_status(status_file.stem))
        return {"jobs": jobs}

    def _resolve_completed_state(self, status: Dict[str, Any]) -> str:
        telemetry_payload = self._load_telemetry_payload(status)
        if telemetry_payload:
            telemetry_status = str(telemetry_payload.get("status", "completed"))
            if telemetry_status in {"completed", "failed"}:
                return telemetry_status
        stderr_content = self._tail_text(Path(status.get("stderr_path", "")))
        return "failed" if stderr_content else "finished"

    def _status_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _write_status(self, job_id: str, payload: Dict[str, Any]) -> None:
        payload = {**payload, "updated_at": datetime.utcnow().isoformat() + "Z"}
        self._status_path(job_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _read_status(self, job_id: str) -> Dict[str, Any]:
        return json.loads(self._status_path(job_id).read_text(encoding="utf-8"))

    def _augment_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        telemetry_payload = self._load_telemetry_payload(status)
        augmented = dict(status)
        if telemetry_payload:
            stage_events = telemetry_payload.get("stage_events", [])
            augmented["progress"] = self._build_progress(stage_events)
            augmented["telemetry_status"] = telemetry_payload.get("status")
            augmented["telemetry_path"] = telemetry_payload.get("_path")
        else:
            augmented["progress"] = {
                "completed_stages": 0,
                "total_stages": 6,
                "latest_stage": None,
                "percentage": 0.0,
            }

        augmented["stdout_tail"] = self._tail_text(Path(augmented["stdout_path"]))
        augmented["stderr_tail"] = self._tail_text(Path(augmented["stderr_path"]))
        return augmented

    def _build_progress(self, stage_events: Any) -> Dict[str, Any]:
        completed = [event for event in stage_events if event.get("status") == "completed"]
        latest_stage = stage_events[-1].get("stage") if stage_events else None
        total_stages = max(6, len(stage_events))
        percentage = round((len(completed) / total_stages) * 100, 1) if total_stages else 0.0
        return {
            "completed_stages": len(completed),
            "total_stages": total_stages,
            "latest_stage": latest_stage,
            "percentage": percentage,
        }

    def _load_telemetry_payload(self, status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        run_dir = Path(status.get("run_dir", ""))
        telemetry_dir = run_dir / "telemetry"
        telemetry_files = sorted(telemetry_dir.glob("pipeline_run_*.json"))
        if not telemetry_files:
            return None
        payload = json.loads(telemetry_files[-1].read_text(encoding="utf-8"))
        payload["_path"] = str(telemetry_files[-1])
        return payload

    def _tail_text(self, path: Path, max_chars: int = 4000) -> str:
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8", errors="replace")
        return content[-max_chars:]