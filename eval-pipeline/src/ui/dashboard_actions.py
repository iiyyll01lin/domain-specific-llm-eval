from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from .dashboard_job_runner import DashboardJobRunner


def build_pipeline_command(
    docs: int,
    samples: int,
    config_path: Optional[str] = None,
) -> List[str]:
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
    return command


def create_dashboard_job(
    docs: int,
    samples: int,
    config_path: Optional[str] = None,
    base_dir: Optional[Path] = None,
):
    runner = DashboardJobRunner(base_dir or Path(__file__).resolve().parents[2])
    return runner.create_job(docs=docs, samples=samples, config_path=config_path)


def get_dashboard_job_status(job_id: str, base_dir: Optional[Path] = None):
    runner = DashboardJobRunner(base_dir or Path(__file__).resolve().parents[2])
    return runner.get_job_status(job_id)


def list_dashboard_jobs(base_dir: Optional[Path] = None):
    runner = DashboardJobRunner(base_dir or Path(__file__).resolve().parents[2])
    return runner.list_jobs()