from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parents[1]


def test_pipeline_help_includes_maintained_cli_options() -> None:
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        cwd=PIPELINE_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--stage" in result.stdout


def test_strategy_templates_contain_maintained_presets() -> None:
    config_path = PIPELINE_DIR / "config" / "testset_strategies.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert "quick_test" in config
    assert "rag_evaluation" in config
    assert "production_ready" in config
    assert config["quick_test"]["testset_generation"]["method"] == "configurable"