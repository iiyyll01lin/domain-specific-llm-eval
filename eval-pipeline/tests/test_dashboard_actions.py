from src.ui.dashboard_actions import build_pipeline_command


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