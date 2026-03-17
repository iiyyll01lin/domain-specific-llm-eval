from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.hybrid_testset_generator import HybridTestsetGenerator
from src.pipeline.config_manager import ConfigManager
from src.ui.force_graph_viewer import ForceGraphVisualizer
from src.utils.ragas_fixes import fix_ragas_dataset_schema


def test_legacy_test_ragas_fix_behaviour_is_covered() -> None:
    ragas_testset_df = pd.DataFrame(
        {
            "user_input": ["What is machine learning?"],
            "reference_contexts": [["ML is a subset of AI"]],
            "response": ["ML is about learning from data"],
            "reference": ["Standard ML definition"],
        }
    )

    fixed = fix_ragas_dataset_schema(ragas_testset_df)
    generator = HybridTestsetGenerator(
        {
            "method": "hybrid",
            "samples_per_document": 1,
            "max_total_samples": 2,
            "testset_generation": {
                "ragas_config": {
                    "custom_llm": {"endpoint": "http://example.invalid", "model": "gpt-4o"}
                }
            },
        }
    )

    secrets_path = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/config/secrets.yaml")
    secrets = yaml.safe_load(secrets_path.read_text(encoding="utf-8"))

    assert "question" in fixed.columns
    assert "answer" in fixed.columns
    assert "ground_truth" in fixed.columns
    assert fixed.iloc[0]["question"] == "What is machine learning?"
    assert generator.method == "hybrid"
    assert str(secrets["inventec_llm"]["api_key"]).startswith("sk-")


def test_legacy_pure_ragas_implementation_behaviour_is_covered() -> None:
    config_path = "/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    manager = ConfigManager(config_path)
    config = manager.load_config()
    csv_path = Path(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/data/csv/pre-training-data.csv"
    )
    df = pd.read_csv(csv_path)
    content = json.loads(str(df.iloc[0]["content"]))

    assert config["testset_generation"]["ragas_config"]["custom_llm"]["endpoint"]
    assert content.get("text")
    assert content.get("title") is not None


def test_legacy_pipeline_integration_expectations_have_maintained_coverage(tmp_path: Path) -> None:
    kg_path = tmp_path / "kg.json"
    kg_path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "relationships": [{"source": "A", "target": "B"}],
            }
        ),
        encoding="utf-8",
    )

    visualizer = ForceGraphVisualizer()
    exported = visualizer.export_from_kg_artifact(kg_path, tmp_path / "topology")
    payload = json.loads(Path(exported["payload_path"]).read_text(encoding="utf-8"))

    assert payload["node_count"] == 3
    assert payload["link_count"] == 1
    assert payload["isolated_nodes"] == ["C"]
    assert payload["high_centrality_nodes"]