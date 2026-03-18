from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from reports.report_generator import ReportGenerator
from src.pipeline.config_manager import ConfigManager


SCRIPT_DIR = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/scripts")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tiktoken_fallback import patch_tiktoken_with_fallback

import test_comprehensive_fixes
import test_config_check
import test_custom_documents
import test_document_chunking
import test_orchestrator_update
import test_report_generation


def test_legacy_report_fixes_behaviour_is_covered(tmp_path: Path) -> None:
    sample_evaluation_data = {
        "rag_results": [
            {
                "question": "What is concept according to the document?",
                "answer": "According to the document, concept is comprehensively explained...",
                "auto_keywords": ["based", "according", "information"],
                "context_precision": 0.5,
                "context_recall": 0.88,
                "faithfulness": 0.826,
                "answer_relevancy": 0.634,
                "kw_metric": 0.508,
                "weighted_average_score": 0.71,
                "keyword_score": 0.833,
                "source_file": "sample.csv",
            }
        ]
    }
    data_file = tmp_path / "evaluation.json"
    data_file.write_text(json.dumps(sample_evaluation_data), encoding="utf-8")

    generator = ReportGenerator({"reporting": {"enabled": True}})
    reports = generator.generate_reports([data_file], "legacy-report-fixes")

    assert reports
    assert any(str(path).endswith(".xlsx") or str(path).endswith(".html") for path in reports.values())


def test_legacy_tiktoken_patch_behaviour_is_covered(monkeypatch) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_ONLY", "1")
    monkeypatch.setenv("TIKTOKEN_DISABLE_DOWNLOAD", "1")
    monkeypatch.setenv("TIKTOKEN_FORCE_OFFLINE", "1")

    patch_tiktoken_with_fallback()

    import tiktoken

    tokenizer = tiktoken.get_encoding("o200k_base")
    tokens = tokenizer.encode("Hello world, this is a test!")

    assert tokens
    assert hasattr(tiktoken, "__spec__")
    assert tiktoken.__spec__ is not None


def test_legacy_full_ragas_implementation_config_smoke_is_covered() -> None:
    config_file = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml")
    manager = ConfigManager(str(config_file))
    config = manager.load_config()

    testset_config = config.get("testset_generation", {})
    ragas_config = testset_config.get("ragas_config", {})
    csv_path = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/data/csv/pre-training-data.csv")
    df = pd.read_csv(csv_path)

    assert config_file.exists()
    assert ragas_config.get("custom_llm", {}).get("endpoint")
    assert not df.empty


def test_legacy_document_chunking_behaviour_is_covered() -> None:
    chunks = test_document_chunking.simple_chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 60,
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_size=30,
    )

    assert len(chunks) >= 2
    assert all(len(chunk) >= 30 for chunk in chunks)


def test_legacy_custom_documents_behaviour_is_covered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    docs_dir = test_custom_documents.create_sample_documents()
    config_file = test_custom_documents.create_test_config(docs_dir)

    assert docs_dir.exists()
    assert len(list(docs_dir.glob("*.txt"))) == 3
    assert Path(config_file).exists()
    assert "custom_data" in Path(config_file).read_text(encoding="utf-8")


def test_legacy_comprehensive_fixes_behaviour_is_covered() -> None:
    config_path = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml")
    config = ConfigManager(str(config_path)).load_config()
    ragas_config = config.get("testset_generation", {}).get("ragas_config", {})
    service_boundary = config.get("evaluation", {}).get("human_feedback", {}).get("service_boundary", {})

    assert config_path.exists()
    assert ragas_config is not None
    assert service_boundary.get("auth", {}).get("api_token") == "local-dev-reviewer-token"


def test_legacy_config_check_behaviour_is_covered() -> None:
    result = test_config_check.check_config_duplication()

    assert result is True


def test_legacy_report_generation_behaviour_is_covered(tmp_path: Path) -> None:
    from src.reports.report_generator import ReportGenerator

    generator = ReportGenerator({"reporting": {"enabled": True}})
    eval_file = tmp_path / "evaluation.json"
    eval_file.write_text(
        json.dumps(
            {
                "rag_results": [
                    {
                        "question": "Q1",
                        "answer": "A1",
                        "context_precision": 0.7,
                        "faithfulness": 0.8,
                        "answer_relevancy": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    reports = generator.generate_reports([eval_file], "legacy-report-generation")

    assert reports


def test_legacy_orchestrator_update_behaviour_is_covered() -> None:
    result = test_orchestrator_update.test_orchestrator_testset_generation()

    assert result["success"] is True
    assert result["testsets_generated"] == 1