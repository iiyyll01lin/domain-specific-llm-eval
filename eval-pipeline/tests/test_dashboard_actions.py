from pathlib import Path

from src.ui.dashboard_actions import (build_pipeline_command,
                                      create_dashboard_job,
                                      get_dashboard_job_status)


def test_build_pipeline_command_includes_required_overrides():
    command = build_pipeline_command(docs=7, samples=9)

    assert command[1] == "run_pure_ragas_pipeline.py"
    assert "--docs" in command
    assert "7" in command
    assert "--samples" in command
    assert "9" in command


def test_build_pipeline_command_includes_optional_config():
    command = build_pipeline_command(docs=3, samples=4, config_path="config/pipeline.yaml")

    assert command[-2] == "--config"
    assert command[-1].endswith("config/pipeline.yaml")


def test_dashboard_job_lifecycle(monkeypatch, tmp_path: Path):
    class FakePopen:
        def __init__(self, command, cwd, stdout, stderr):
            self.command = command
            self.cwd = cwd
            self.pid = 4321

    monkeypatch.setattr("src.ui.dashboard_job_runner.subprocess.Popen", FakePopen)
    monkeypatch.setattr("src.ui.dashboard_job_runner.Path.exists", lambda self: self.as_posix() == "/proc/4321")

    job = create_dashboard_job(docs=2, samples=3, base_dir=tmp_path)

    assert job["status"] == "running"
    status = get_dashboard_job_status(job["job_id"], base_dir=tmp_path)
    assert status["status"] == "running"