from __future__ import annotations

import json
import subprocess
import sys
import uuid
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
        }
        self._write_status(job_id, status)

        process = subprocess.Popen(
            command,
            cwd=str(self.base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status.update({"status": "running", "pid": process.pid})
        self._write_status(job_id, status)
        return status

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        status = self._read_status(job_id)
        pid = status.get("pid")
        if status.get("status") != "running" or not isinstance(pid, int):
            return status

        proc_path = Path("/proc") / str(pid)
        if proc_path.exists():
            return status

        updated_status = {**status, "status": self._resolve_completed_state(job_id)}
        self._write_status(job_id, updated_status)
        return updated_status

    def list_jobs(self) -> Dict[str, Any]:
        jobs = []
        for status_file in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            jobs.append(self.get_job_status(status_file.stem))
        return {"jobs": jobs}

    def _resolve_completed_state(self, job_id: str) -> str:
        log_files = list((self.base_dir / "outputs").glob("pure_ragas_run_*/telemetry/pipeline_run_*.json"))
        return "completed" if log_files else "finished"

    def _status_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _write_status(self, job_id: str, payload: Dict[str, Any]) -> None:
        self._status_path(job_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _read_status(self, job_id: str) -> Dict[str, Any]:
        return json.loads(self._status_path(job_id).read_text(encoding="utf-8"))