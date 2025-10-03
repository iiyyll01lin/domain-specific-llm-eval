from __future__ import annotations

import json
from pathlib import Path

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers.testset_schema import TestsetSample

from services.eval.execution import execute_evaluation_run
from services.eval.metrics.loader import load_metric_registry
from services.eval.persistence_pipeline import EvaluationPersistencePipeline
from services.eval.rag_interface import RetrievedContext, StaticResponseAdapter
from services.eval.runner import EvaluationRunner


def _build_sample() -> TestsetSample:
    return TestsetSample(
        eval_sample=SingleTurnSample(
            user_input="Where is the Taipei 101 located?",
            reference="The Taipei 101 is located in Taipei, Taiwan.",
        ),
        synthesizer_name="unit-test",
    )


def test_execute_evaluation_run_persists_metrics(tmp_path: Path) -> None:
    registry = load_metric_registry(entry_point_group=None)
    adapter = StaticResponseAdapter(
        answer="The Taipei 101 is located in Taipei, Taiwan.",
        contexts=(
            RetrievedContext(
                text="Taipei 101 is a landmark skyscraper located in Taipei, Taiwan.",
                document_id="chunk-1",
                score=0.92,
                metadata={"source": "unit"},
            ),
        ),
        latency_ms=15.0,
    )
    runner = EvaluationRunner(adapter=adapter)
    pipeline = EvaluationPersistencePipeline("run-xyz", tmp_path, flush_interval_seconds=0.0)

    summary = execute_evaluation_run(
        run_id="run-xyz",
        samples=[_build_sample()],
        runner=runner,
        metric_registry=registry,
        pipeline=pipeline,
        timeout_seconds=30.0,
        max_retries=1,
    )

    assert summary.run_id == "run-xyz"
    assert summary.item_count == 1
    assert "faithfulness" in summary.metric_versions

    artifacts = summary.artifacts
    assert set(artifacts.keys()) == {"items", "manifest", "kpis"}

    item_lines = artifacts["items"].read_text(encoding="utf-8").strip().splitlines()
    assert len(item_lines) == 1
    payload = json.loads(item_lines[0])

    metrics = payload["metadata"]["metrics"]
    assert "faithfulness" in metrics
    assert metrics["faithfulness"] >= 0.8
    assert metrics["answer_relevancy"] > 0
    assert "metric_details" in payload["metadata"]
    assert "faithfulness" in payload["metadata"]["metric_details"]

    aggregation = summary.aggregation
    assert aggregation.counts["records"] == 1
    assert "faithfulness" in aggregation.metrics
    assert aggregation.metrics["faithfulness"].average >= 0.8
