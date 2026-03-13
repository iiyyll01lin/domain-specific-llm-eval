from __future__ import annotations

from pathlib import Path

import pandas as pd

from reports.report_generator import ReportGenerator
from data.configurable_testset_builder import ConfigurableTestsetBuilder
from pipeline.config_manager import ConfigManager


def test_report_generator_maps_required_columns_without_crashing(tmp_path: Path) -> None:
    generator = ReportGenerator({"reporting": {"enabled": True}})
    results_df = pd.DataFrame(
        {
            "context_precision": [0.9],
            "faithfulness": [0.8],
            "answer_relevancy": [0.85],
            "question": ["What changed?"],
            "rag_answer": ["Normalized answer"],
            "source_file": ["sample.csv"],
        }
    )

    reports = generator.generate_comprehensive_report(
        evaluation_results=results_df,
        evaluation_summary={"run_id": "test-run", "timestamp": "2026-03-13"},
        output_dir=tmp_path,
    )

    assert "detailed_results" in reports
    assert (tmp_path / "detailed_results.xlsx").exists()
    assert "overall_score" in results_df.columns
    assert "ragas_composite_score" in results_df.columns


def test_configurable_builder_and_config_manager_smoke() -> None:
    builder = ConfigurableTestsetBuilder(
        config_path="/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    )
    assert "synthetic_llm" in builder.available_strategies
    assert "factual" in builder.question_types

    manager = ConfigManager(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    )
    config = manager.load_config()
    assert "testset_generation" in config
    assert "rag_system" in config
