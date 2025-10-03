from __future__ import annotations

import json
from pathlib import Path

from prometheus_client import CollectorRegistry

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.persistence_pipeline import EvaluationPersistencePipeline
from services.eval.rag_interface import RetrievedContext
from services.eval.persistence_metrics import PersistenceMetricsRecorder
from services.eval.aggregation_metrics import AggregationMetricsRecorder


def _make_item(sample_id: str, success: bool = True) -> CapturedEvaluationItem:
    return CapturedEvaluationItem(
        run_id="run-xyz",
        sample_id=sample_id,
        question="Question?",
        answer="Answer",
        contexts=(RetrievedContext(text="Context"),),
        success=success,
        metadata={},
        raw={},
    )


def test_pipeline_persists_items_and_kpis(tmp_path: Path) -> None:
    persistence_registry = CollectorRegistry()
    aggregation_registry = CollectorRegistry()
    pipeline = EvaluationPersistencePipeline(
        "run-xyz",
        tmp_path,
        flush_interval_seconds=0.0,
        persistence_metrics=PersistenceMetricsRecorder(registry=persistence_registry),
        aggregation_metrics=AggregationMetricsRecorder(registry=aggregation_registry),
    )

    pipeline.submit(_make_item("sample-1"), {"faithfulness": 0.9, "answer_relevancy": 0.8})
    pipeline.submit(_make_item("sample-2"), {"faithfulness": 0.95, "answer_relevancy": None})

    pipeline.wait_until_drained(timeout=1.0)
    result = pipeline.finalize()

    artifacts = pipeline.artifacts()
    assert artifacts["items"].exists()
    assert artifacts["manifest"].exists()
    assert artifacts["kpis"].exists()

    items_content = artifacts["items"].read_text(encoding="utf-8").strip().splitlines()
    assert len(items_content) == 2

    kpis_payload = json.loads(artifacts["kpis"].read_text(encoding="utf-8"))
    assert kpis_payload["run_id"] == "run-xyz"
    assert kpis_payload["counts"]["records"] == 2
    assert "faithfulness" in kpis_payload["metrics"]

    assert result.metrics["faithfulness"].count == 2


def test_pipeline_rejects_submissions_after_finalize(tmp_path: Path) -> None:
    pipeline = EvaluationPersistencePipeline("run-abc", tmp_path, flush_interval_seconds=0.0)
    pipeline.submit(_make_item("sample-1"), {"faithfulness": 0.9})
    pipeline.finalize()

    try:
        pipeline.submit(_make_item("sample-2"), {"faithfulness": 0.95})
        assert False
    except RuntimeError:
        pass


def test_pipeline_finalize_twice_returns_cached_result(tmp_path: Path) -> None:
    pipeline = EvaluationPersistencePipeline("run-abc", tmp_path, flush_interval_seconds=0.0)
    pipeline.submit(_make_item("sample-1"), {"faithfulness": 0.9})
    first = pipeline.finalize()
    second = pipeline.finalize()
    assert first is second
