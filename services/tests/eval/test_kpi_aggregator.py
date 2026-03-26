from __future__ import annotations

import math

from prometheus_client import CollectorRegistry

from services.eval.aggregation.aggregator import KPIAggregator
from services.eval.aggregation_metrics import AggregationMetricsRecorder


def test_kpi_aggregator_computes_distributions() -> None:
    registry = CollectorRegistry()
    aggregator = KPIAggregator(metrics_recorder=AggregationMetricsRecorder(registry=registry))
    records = [
        {"faithfulness": 0.9, "answer_relevancy": 0.8},
        {"faithfulness": 0.95, "answer_relevancy": 0.85},
        {"faithfulness": 0.92, "answer_relevancy": None},
    ]

    result = aggregator.aggregate(records)

    assert result.counts["records"] == 3
    assert result.counts["metrics"] == 2
    assert "faithfulness" in result.metrics
    faithfulness = result.metrics["faithfulness"]
    assert faithfulness.count == 3
    assert math.isclose(faithfulness.minimum, 0.9)
    assert math.isclose(faithfulness.maximum, 0.95)
    assert "answer_relevancy" in result.metrics
    answer_rel = result.metrics["answer_relevancy"]
    assert answer_rel.count == 2

    records_total = registry.get_sample_value("eval_aggregation_records_total")
    assert records_total == 3.0
    metrics_total = registry.get_sample_value("eval_aggregation_metrics_total")
    assert metrics_total == 2.0
    duration_count = registry.get_sample_value("eval_aggregation_duration_seconds_count")
    assert duration_count == 1.0


def test_kpi_aggregator_skips_all_missing_metric() -> None:
    aggregator = KPIAggregator(metrics_recorder=AggregationMetricsRecorder(registry=CollectorRegistry()))
    records = [
        {"faithfulness": None},
        {"faithfulness": float("nan")},
    ]

    result = aggregator.aggregate(records)
    assert result.metrics == {}
    assert result.counts["missing_values"] == 2
