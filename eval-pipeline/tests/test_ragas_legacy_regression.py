from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.pipeline.config_manager import ConfigManager
# PipelineOrchestrator is imported lazily inside tests to avoid importlib hang at collection time


def test_field_mapping_fix_normalizes_ragas_columns() -> None:
    ragas_testset_df = pd.DataFrame(
        {
            "user_input": ["What is machine learning?"],
            "reference_contexts": [["ML is a subset of AI"]],
            "response": ["ML is about learning from data"],
            "reference": ["Standard ML definition"],
        }
    )

    field_mapping = {
        "user_input": "question",
        "reference_contexts": "contexts",
        "response": "answer",
        "reference": "ground_truth",
    }
    normalized = ragas_testset_df.rename(columns=field_mapping)
    normalized["contexts"] = normalized["contexts"].apply(
        lambda value: value[0] if isinstance(value, list) and value else ""
    )

    assert list(normalized.columns) == ["question", "contexts", "answer", "ground_truth"]
    assert normalized.iloc[0]["contexts"] == "ML is a subset of AI"


def test_pipeline_config_and_secrets_smoke() -> None:
    config_path = "/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    manager = ConfigManager(config_path)
    config = manager.load_config()

    assert "testset_generation" in config
    assert "rag_system" in config
    assert config["testset_generation"]["ragas_config"]["custom_llm"]["endpoint"]

    secrets_path = Path(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/config/secrets.yaml"
    )
    secrets = yaml.safe_load(secrets_path.read_text(encoding="utf-8"))
    api_key = str(secrets["inventec_llm"]["api_key"])

    assert api_key.startswith("sk-")
    assert "placeholder" not in api_key


def test_hybrid_generator_and_ragas_imports_smoke() -> None:
    from src.data.hybrid_testset_generator import HybridTestsetGenerator
    from ragas import evaluate as ragas_evaluate  # type: ignore[attr-defined]
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.testset import TestsetGenerator

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

    assert generator.method == "hybrid"
    assert TestsetGenerator is not None
    assert ragas_evaluate is not None
    assert answer_relevancy is not None
    assert context_precision is not None
    assert faithfulness is not None


def test_pure_ragas_components_and_csv_loading_smoke() -> None:
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType

    csv_path = Path(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/data/csv/pre-training-data.csv"
    )
    df = pd.read_csv(csv_path)
    content = json.loads(str(df.iloc[0]["content"]))

    kg = KnowledgeGraph()
    node = Node(
        id="12345678-1234-5678-1234-567812345678",
        type=NodeType.DOCUMENT,
        properties={"content": content.get("text", ""), "title": content.get("title", "")},
    )
    kg._add_node(node)

    assert LangchainLLMWrapper is not None
    assert len(kg.nodes) == 1
    assert content.get("text")


def test_orchestrator_hyperparameter_helper_runs_when_enabled(tmp_path: Path) -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: avoids importlib hang at collection
    from src.optimization.hyperparam_search import OptunaOptimizer

    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.hyperparameter_optimizer = OptunaOptimizer(
        n_trials=3, output_dir=str(tmp_path)
    )

    result = orchestrator._run_hyperparameter_search({"success_rate": 0.7})

    assert result is not None
    assert result["trial_count"] == 3
    assert (tmp_path / "trial_history.json").exists()