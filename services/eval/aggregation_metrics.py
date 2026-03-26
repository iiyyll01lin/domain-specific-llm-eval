from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram


class AggregationMetricsRecorder:
    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self._duration_histogram = Histogram(
            "eval_aggregation_duration_seconds",
            "Duration of KPI aggregation runs",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=registry,
        )
        self._records_counter = Counter(
            "eval_aggregation_records_total",
            "Total number of evaluation records processed during aggregation",
            registry=registry,
        )
        self._metrics_counter = Counter(
            "eval_aggregation_metrics_total",
            "Total number of metric distributions produced",
            registry=registry,
        )

    def record(self, *, item_count: int, metric_count: int, duration_seconds: float) -> None:
        self._duration_histogram.observe(duration_seconds)
        self._records_counter.inc(item_count)
        self._metrics_counter.inc(metric_count)


__all__ = ["AggregationMetricsRecorder"]
