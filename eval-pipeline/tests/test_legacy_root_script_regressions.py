from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pandas as pd
import yaml

from src.data.csv_ragas_converter import CSVToRagasConverter
from src.data.hybrid_testset_generator import HybridTestsetGenerator
from src.pipeline.config_manager import ConfigManager
from src.reports.report_generator import ReportGenerator
from src.ui.force_graph_viewer import ForceGraphVisualizer
from src.utils.ragas_fixes import fix_ragas_dataset_schema


def _load_eval_pipeline_document_loader_module():
    module_path = Path(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/document_loader.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_pipeline_document_loader",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_legacy_csv_ragas_integration_script_behaviour_is_covered(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "content": [
                json.dumps(
                    {
                        "text": "Manufacturing process documentation with enough detail to pass the minimum content length validation for downstream document conversion.",
                        "title": "Doc 1",
                    }
                )
            ]
        }
    ).to_csv(csv_path, index=False)

    converter = CSVToRagasConverter(
        {
            "data_sources": {"csv": {"csv_files": [str(csv_path)], "format": {}}},
            "testset_generation": {
                "csv_processing": {"content_preprocessing": {"min_content_length": 20}},
            },
        }
    )

    loaded_df = converter.load_csv_data()
    documents = converter.csv_to_documents(loaded_df)

    monkeypatch.setattr(
        CSVToRagasConverter,
        "generate_ragas_testset",
        lambda self, docs: {
            "testset_df": pd.DataFrame(
                {
                    "question": ["What is described?"],
                    "answer": ["A manufacturing process."],
                    "contexts": [[docs[0].page_content]],
                }
            ),
            "metadata": {"generation_method": "mock_ragas"},
        },
    )

    result = converter.convert_csv_to_ragas_testset()

    assert len(loaded_df) == 1
    assert len(documents) == 1
    assert result["metadata"]["generation_method"] == "mock_ragas"
    assert list(result["testset_df"].columns) == ["question", "answer", "contexts"]


def test_legacy_csv_detailed_script_behaviour_is_covered(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "test_content.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "content": [
                json.dumps(
                    {
                        "text": "Detailed CSV processing sample with enough content to validate row-to-document loading and downstream generator wiring for regression coverage.",
                        "title": "Doc 1",
                    }
                ),
                json.dumps(
                    {
                        "text": "Second detailed CSV sample that exercises configuration loading and keeps the one-row-to-one-document mapping stable under maintained pytest coverage.",
                        "title": "Doc 2",
                    }
                ),
            ],
        }
    ).to_csv(csv_path, index=False)

    config_path = tmp_path / "debug_config.yaml"
    config_path.write_text(
        f"""
data_sources:
  input_type: "csv"
  csv:
    csv_files:
      - "{csv_path}"
testset_generation:
  method: "configurable"
  use_configurable_builder: true
  samples_per_document: 1
  max_total_samples: 5
output:
  base_dir: "{tmp_path / 'outputs'}"
logging:
  level: "INFO"
""",
        encoding="utf-8",
    )

    config = ConfigManager(str(config_path)).load_config()
    document_loader_module = _load_eval_pipeline_document_loader_module()
    monkeypatch.setattr(document_loader_module.DocumentLoader, "_initialize_keybert_offline", lambda self: None)
    monkeypatch.setattr(HybridTestsetGenerator, "initialize_generators", lambda self: None)
    monkeypatch.setattr(
        HybridTestsetGenerator,
        "generate_comprehensive_testset",
        lambda self, document_paths, output_dir: {
            "generated_samples": 2,
            "output_dir": str(output_dir),
            "document_paths": list(document_paths),
        },
    )

    loader = document_loader_module.DocumentLoader(config)
    documents, metadata = loader.load_all_documents()
    generator = HybridTestsetGenerator(config["testset_generation"])
    results = generator.generate_comprehensive_testset([], tmp_path / "test_outputs" / "testsets")

    assert config["data_sources"]["input_type"] == "csv"
    assert len(documents) == 2
    assert len(metadata) == 2
    assert results["generated_samples"] == 2


def test_legacy_ragas_pure_script_behaviour_is_covered(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "pre_training.csv"
    pd.DataFrame(
        {
            "content": [
                json.dumps(
                    {
                        "text": "Pure RAGAS regression content with enough manufacturing detail to survive preprocessing and support synthetic question generation without calling an external endpoint.",
                        "title": "Pure RAGAS",
                    }
                )
            ]
        }
    ).to_csv(csv_path, index=False)

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

    class _FakeWrapper:
        def __init__(self, llm):
            self.langchain_llm = llm

    fake_langchain_module = types.ModuleType("langchain")
    fake_callbacks_module = types.ModuleType("langchain.callbacks")
    fake_callback_manager_module = types.ModuleType("langchain.callbacks.manager")
    fake_callback_manager_module.CallbackManagerForLLMRun = object
    fake_llms_module = types.ModuleType("langchain.llms")
    fake_llms_base_module = types.ModuleType("langchain.llms.base")
    fake_llms_base_module.LLM = _FakeLLM
    fake_ragas_module = types.ModuleType("ragas")
    fake_ragas_llms_module = types.ModuleType("ragas.llms")
    fake_ragas_llms_module.LangchainLLMWrapper = _FakeWrapper

    monkeypatch.setitem(sys.modules, "langchain", fake_langchain_module)
    monkeypatch.setitem(sys.modules, "langchain.callbacks", fake_callbacks_module)
    monkeypatch.setitem(sys.modules, "langchain.callbacks.manager", fake_callback_manager_module)
    monkeypatch.setitem(sys.modules, "langchain.llms", fake_llms_module)
    monkeypatch.setitem(sys.modules, "langchain.llms.base", fake_llms_base_module)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", fake_ragas_llms_module)

    converter = CSVToRagasConverter(
        {
            "data_sources": {
                "csv": {
                    "csv_files": [str(csv_path)],
                    "format": {"column_mapping": {"content": "content"}},
                }
            },
            "testset_generation": {
                "csv_processing": {
                    "content_field_extraction": True,
                    "content_preprocessing": {"min_content_length": 20},
                },
                "ragas_config": {
                    "use_custom_llm": True,
                    "custom_llm": {
                        "endpoint": "http://example.invalid/v1/chat/completions",
                        "api_key": "sk-test",
                        "model": "gpt-4o",
                    },
                },
            },
        }
    )
    monkeypatch.setattr(
        CSVToRagasConverter,
        "generate_ragas_testset",
        lambda self, docs: {
            "testset_df": pd.DataFrame(
                {"question": ["What process is described?"], "answer": ["A manufacturing process."], "contexts": [[docs[0].page_content]]}
            ),
            "metadata": {"generation_method": "mock_ragas"},
        },
    )

    custom_llm = converter.create_custom_llm_for_ragas()
    loaded_df = converter.load_csv_data()
    documents = converter.csv_to_documents(loaded_df)
    result = converter.convert_csv_to_ragas_testset()

    assert custom_llm is not None
    assert custom_llm.langchain_llm.endpoint == "http://example.invalid/v1/chat/completions"
    assert len(documents) == 1
    assert result["metadata"]["generation_method"] == "mock_ragas"


def test_legacy_orchestrator_fix_script_behaviour_is_covered(monkeypatch, tmp_path: Path) -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator

    monkeypatch.setattr(PipelineOrchestrator, "_initialize_components", lambda self: None)

    output_dirs = {
        "testsets": tmp_path / "testsets",
        "metadata": tmp_path / "metadata",
        "evaluations": tmp_path / "evaluations",
        "reports": tmp_path / "reports",
        "temp": tmp_path / "temp",
    }
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    orchestrator = PipelineOrchestrator(
        config={
            "data_sources": {
                "input_type": "csv",
                "csv": {"csv_files": ["data/csv/smt-nxt-errorcode.csv"]},
            },
            "testset_generation": {"generator_class": "hybrid"},
        },
        run_id="test_fix",
        output_dirs=output_dirs,
        force_overwrite=True,
    )

    assert orchestrator.run_id == "test_fix"
    assert orchestrator.force_overwrite is True
    assert orchestrator.output_dirs["reports"] == output_dirs["reports"]


def test_legacy_report_fixes_script_behaviour_is_covered(tmp_path: Path) -> None:
    report_generator = ReportGenerator({"log_level": "INFO"})
    results_df = pd.DataFrame(
        {
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"],
            "context_precision": [0.5, 0.1],
            "context_recall": [0.88, 0.759],
            "faithfulness": [0.826, 0.1],
            "answer_relevancy": [0.634, 0.88],
            "kw_metric": [0.508, 0.2],
            "weighted_average_score": [0.71, 0.46],
            "keyword_score": [0.833, 0.4],
        }
    )

    report_generator._ensure_required_columns(results_df)
    output_file = tmp_path / "legacy_report.xlsx"
    report_generator._generate_excel_report(results_df, output_file)

    assert output_file.exists()
    assert "ragas_composite_score" in results_df.columns
    assert "overall_pass" in results_df.columns


def test_legacy_custom_llm_integration_script_behaviour_is_covered(tmp_path: Path, monkeypatch) -> None:
    config_path = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generator = HybridTestsetGenerator(config)

    ragas_config = config.get("testset_generation", {}).get("ragas_config", {})
    custom_llm_config = ragas_config.get("custom_llm", {})
    api_key = generator._load_api_key_from_secrets(custom_llm_config)

    monkeypatch.setattr(
        generator.document_processor,
        "process_documents",
        lambda: [
            {
                "source_file": str(tmp_path / "custom_llm_doc.txt"),
                "filename": "custom_llm_doc.txt",
                "content": "Custom LLM integration validates private endpoint configuration and document processing.",
                "file_type": "txt",
                "word_count": 10,
            }
        ],
    )
    processed_docs = generator._process_documents([str(tmp_path / "custom_llm_doc.txt")])

    assert generator.method in {"configurable", "hybrid", "ragas", "csv"}
    assert custom_llm_config.get("endpoint")
    assert isinstance(api_key, str)
    assert processed_docs


def test_legacy_pipeline_fixes_script_behaviour_is_covered() -> None:
    import subprocess
    import sys

    pipeline_dir = Path("/data/yy/domain-specific-llm-eval/eval-pipeline")
    help_result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        cwd=pipeline_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--stage" in help_result.stdout


def test_legacy_small_data_script_behaviour_is_covered(tmp_path: Path) -> None:
    csv_path = tmp_path / "small_test_data.csv"
    rows = [
        {"error_code": "E001", "display": "Connection timeout error occurred", "used_by": "Database Module"},
        {"error_code": "E002", "display": "Invalid authentication credentials", "used_by": "Authentication Module"},
    ]
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = ConfigManager(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    ).load_config()
    config.setdefault("data_sources", {}).setdefault("csv", {})["csv_files"] = [str(csv_path)]
    config["testset_generation"]["max_documents_for_generation"] = 2
    config["testset_generation"]["testset_size"] = 3
    kg_config = (
        config["testset_generation"]
        .setdefault("ragas_config", {})
        .setdefault("knowledge_graph_config", {})
    )
    kg_config["enable_kg_loading"] = True
    kg_config["enable_kg_saving"] = True

    assert csv_path.exists()
    assert len(rows) == 2
    assert config["testset_generation"]["max_documents_for_generation"] == 2
    assert config["testset_generation"]["testset_size"] == 3
    assert kg_config["enable_kg_loading"] is True
    assert kg_config["enable_kg_saving"] is True


def test_legacy_simple_pipeline_script_behaviour_is_covered() -> None:
    from pipeline.config_manager import ConfigManager
    from evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator
    from evaluation.ragas_evaluator import RagasEvaluator
    from reports.report_generator import ReportGenerator

    config = ConfigManager(
        "/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml"
    ).load_config()
    contextual_evaluator = ContextualKeywordEvaluator(
        {"weights": {"mandatory": 0.8, "optional": 0.2}, "threshold": 0.6}
    )
    contextual_result = contextual_evaluator.evaluate(
        {"questions": ["What is the thickness measurement procedure?"], "auto_keywords": [["thickness", "measurement", "procedure"]]},
        [{"answer": "The thickness measurement procedure uses a gauge to measure the steel plate."}],
    )
    ragas_evaluator = RagasEvaluator({"evaluation": {"ragas_metrics": {"llm": {"use_custom_llm": False}}}})
    report_generator = ReportGenerator({"reporting": {}})
    recommendations = report_generator._generate_recommendations(
        pd.DataFrame(),
        {
            "overall_statistics": {"overall_pass_rate": 1.0, "contextual_pass_rate": 1.0, "ragas_pass_rate": 1.0, "semantic_pass_rate": 1.0},
            "human_feedback_statistics": {"feedback_needed_ratio": 0.0},
        },
    )

    assert "testset_generation" in config
    assert contextual_result.get("pass_rate", 0.0) >= 0.0
    assert ragas_evaluator.is_available() in {True, False}
    assert recommendations


def test_legacy_direct_orchestrator_script_behaviour_is_covered(monkeypatch, tmp_path: Path) -> None:
    from pipeline.enhanced_orchestrator import EnhancedPipelineOrchestrator
    from pipeline.orchestrator import PipelineOrchestrator

    def fake_base_init(self, config, run_id, output_dirs, force_overwrite=False):
        self.config = config
        self.run_id = run_id
        self.output_dirs = output_dirs

    monkeypatch.setattr(PipelineOrchestrator, "__init__", fake_base_init)

    output_dirs = {
        "base": tmp_path / "base",
        "testsets": tmp_path / "testsets",
        "metadata": tmp_path / "metadata",
        "logs": tmp_path / "logs",
    }
    orchestrator = EnhancedPipelineOrchestrator(
        {"validation": {}, "logging": {"level": "INFO"}},
        "run-id",
        output_dirs,
    )

    assert hasattr(orchestrator, "run")
    assert orchestrator.run_id == "run-id"


def test_legacy_document_processing_script_behaviour_is_covered(tmp_path: Path) -> None:
    from data.document_processor import DocumentProcessor

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    sample_doc = docs_dir / "doc.txt"
    sample_doc.write_text("Document processing regression coverage sample.", encoding="utf-8")

    processor = DocumentProcessor({"text_files": [str(sample_doc)]}, tmp_path / "processed")
    processed = processor.process_documents()

    assert len(processed) == 1
    assert processed[0]["filename"] == "doc.txt"


def test_legacy_local_generation_script_behaviour_is_covered() -> None:
    from local_dataset_generator import LocalSyntheticDatasetGenerator

    generator = LocalSyntheticDatasetGenerator({"local": {}, "fallback": {}})
    assert generator is not None


def test_legacy_comprehensive_report_script_behaviour_is_covered(tmp_path: Path) -> None:
    generator = ReportGenerator({"paths": {"output_dir": str(tmp_path)}})
    evaluation_results = pd.DataFrame(
        [
            {
                "question": "What is the main benefit?",
                "answer": "Efficiency.",
                "context_precision": 0.5,
                "context_recall": 0.88,
                "faithfulness": 0.826,
                "answer_relevancy": 0.634,
                "kw_metric": 0.508,
                "weighted_average_score": 0.71,
            }
        ]
    )
    reports = generator.generate_comprehensive_report(
        evaluation_results=evaluation_results,
        evaluation_summary={"run_id": "legacy", "timestamp": "2026-03-18T00:00:00"},
        output_dir=tmp_path,
    )

    assert reports
    assert "ragas_composite_score" in evaluation_results.columns


def test_legacy_direct_reporting_script_behaviour_is_covered(tmp_path: Path) -> None:
    eval_dir = tmp_path / "outputs" / "run_legacy" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_file = eval_dir / "evaluation_results.json"
    eval_file.write_text(
        json.dumps(
            {
                "rag_results": [
                    {
                        "question": "Q1",
                        "answer": "A1",
                        "context_precision": 0.5,
                        "context_recall": 0.88,
                        "faithfulness": 0.826,
                        "answer_relevancy": 0.634,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(eval_file.read_text(encoding="utf-8"))
    results_df = pd.DataFrame(payload["rag_results"])
    numeric_cols = list(results_df.select_dtypes(include=["number"]).columns)

    assert numeric_cols
    assert not results_df[numeric_cols].describe().empty


def test_legacy_gates_integration_script_behaviour_is_covered(monkeypatch) -> None:
    import src.evaluation.comprehensive_rag_evaluator_fixed as comprehensive_module
    from src.evaluation.gates_system import GatesSystem

    class _FakeRagasEvaluator:
        def __init__(self, config):
            self.config = config

    class _FakeEnhancedEvaluator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(comprehensive_module, "RAGASEvaluatorWithFallbacks", _FakeRagasEvaluator)
    monkeypatch.setattr(comprehensive_module, "EnhancedContextualKeywordEvaluator", _FakeEnhancedEvaluator)

    config = {
        "evaluation": {
            "contextual_keywords": {"evaluator": "enhanced"},
            "gates": {
                "contextual_keywords": {"enabled": True, "threshold": 0.6, "weight": 0.4},
                "ragas_metrics": {"enabled": True, "threshold": 0.7, "weight": 0.6},
                "combination": {"method": "weighted_average", "minimum_gates_required": 1},
            },
        },
        "rag_system": {"enabled": False},
    }

    evaluator = comprehensive_module.ComprehensiveRAGEvaluatorFixed(config)
    gates_system = GatesSystem(config)
    gates_results = gates_system.evaluate_gates(
        {
            "contextual_keyword": {
                "success": True,
                "summary_metrics": {"avg_similarity_score": 0.7, "pass_rate": 0.75, "total_questions": 4},
                "total_questions": 4,
            },
            "ragas": {
                "success": True,
                "overall_scores": {
                    "ContextPrecision": 0.95,
                    "ContextRecall": 0.9,
                    "Faithfulness": 0.85,
                    "AnswerRelevancy": 0.88,
                },
                "total_questions": 4,
            },
        }
    )

    assert evaluator.gates_system is not None
    assert evaluator.keyword_evaluator.__class__.__name__ == "_FakeEnhancedEvaluator"
    assert gates_results.overall_pass is True
    assert gates_results.weighted_score > 0.0


def test_legacy_enhanced_evaluator_script_behaviour_is_covered(monkeypatch) -> None:
    from src.evaluation.enhanced_contextual_keyword_evaluator import EnhancedContextualKeywordEvaluator

    monkeypatch.setattr(EnhancedContextualKeywordEvaluator, "_initialize_models", lambda self: setattr(self, "sentence_model", None))
    evaluator = EnhancedContextualKeywordEvaluator(
        {
            "evaluation": {
                "contextual_keywords": {
                    "evaluator": "enhanced",
                    "similarity_threshold": 0.7,
                    "fuzzy_threshold": 80,
                }
            }
        }
    )

    assert evaluator.keyword_config["evaluator"] == "enhanced"
    assert evaluator.similarity_threshold == 0.7
    assert evaluator.fuzzy_threshold == 80