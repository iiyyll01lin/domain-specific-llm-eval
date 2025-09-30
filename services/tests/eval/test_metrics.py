from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Counter, Histogram

from services.eval.metrics import PrometheusRAGRequestMetrics


def test_rag_request_metrics_records_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="services.eval.metrics")
    registry = CollectorRegistry()
    latency = Histogram(
        "test_rag_latency_seconds",
        "test",
        labelnames=("outcome",),
        registry=registry,
    )
    attempts = Counter(
        "test_rag_attempt_total",
        "test",
        labelnames=("outcome",),
        registry=registry,
    )

    metrics = PrometheusRAGRequestMetrics(latency_histogram=latency, attempts_counter=attempts)

    metrics.record(outcome="success", latency_seconds=1.23, attempts=2, trace_id="trace-1")

    # Histogram keeps sum/count per label
    collected = latency.collect()[0].samples
    count_sample = next(sample for sample in collected if sample.name.endswith("_count"))
    sum_sample = next(sample for sample in collected if sample.name.endswith("_sum"))

    assert count_sample.value == 1.0
    assert sum_sample.value == 1.23

    attempt_sample = attempts.collect()[0].samples[0]
    assert attempt_sample.value == 2.0
    assert attempt_sample.labels["outcome"] == "success"

    assert any(record.message == "rag.request_metrics" and record.trace_id == "trace-1" for record in caplog.records)
