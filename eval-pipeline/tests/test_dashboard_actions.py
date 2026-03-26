from pathlib import Path
import json

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
    original_exists = Path.exists

    class FakePopen:
        def __init__(self, command, cwd, stdout, stderr, env):
            self.command = command
            self.cwd = cwd
            self.pid = 4321
            stdout.write("pipeline started\n")
            stdout.flush()
            telemetry_dir = tmp_path / "outputs" / f"pure_ragas_run_{env['PIPELINE_RUN_ID']}" / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            (telemetry_dir / "pipeline_run_test.json").write_text(
                json.dumps(
                    {
                        "status": "in_progress",
                        "stage_events": [
                            {"stage": "configuration", "status": "completed"},
                            {"stage": "document_loading", "status": "completed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr("src.ui.dashboard_job_runner.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "src.ui.dashboard_job_runner.Path.exists",
        lambda self: True if self.as_posix() == "/proc/4321" else original_exists(self),
    )

    job = create_dashboard_job(docs=2, samples=3, base_dir=tmp_path)

    assert job["status"] == "running"
    status = get_dashboard_job_status(job["job_id"], base_dir=tmp_path)
    assert status["status"] == "running"
    assert status["progress"]["completed_stages"] == 2
    assert status["progress"]["latest_stage"] == "document_loading"
    assert "pipeline started" in status["stdout_tail"]


def test_dashboard_job_marks_exited_in_progress_run_as_failed(monkeypatch, tmp_path: Path):
    original_exists = Path.exists

    class FakePopen:
        def __init__(self, command, cwd, stdout, stderr, env):
            self.pid = 9876
            telemetry_dir = tmp_path / "outputs" / f"pure_ragas_run_{env['PIPELINE_RUN_ID']}" / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            (telemetry_dir / "pipeline_run_test.json").write_text(
                json.dumps(
                    {
                        "status": "in_progress",
                        "stage_events": [
                            {"stage": "configuration", "status": "completed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stderr.write("pipeline crashed\n")
            stderr.flush()

    monkeypatch.setattr("src.ui.dashboard_job_runner.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "src.ui.dashboard_job_runner.Path.exists",
        lambda self: False if self.as_posix() == "/proc/9876" else original_exists(self),
    )

    job = create_dashboard_job(docs=1, samples=1, base_dir=tmp_path)
    status = get_dashboard_job_status(job["job_id"], base_dir=tmp_path)

    assert status["status"] == "failed"
    assert "pipeline crashed" in status["stderr_tail"]