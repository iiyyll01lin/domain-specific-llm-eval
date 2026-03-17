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