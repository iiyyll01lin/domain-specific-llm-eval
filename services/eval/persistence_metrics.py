from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram


class PersistenceMetricsRecorder:
    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry
        self._item_counter = Counter(
            "eval_items_written_total",
            "Total evaluation items written",
            registry=registry,
        )
        self._flush_latency = Histogram(
            "eval_item_flush_latency_seconds",
            "Flush latency for evaluation item persistence",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=registry,
        )
        self._flush_bytes = Counter(
            "eval_item_flush_bytes_total",
            "Total bytes written during flush operations",
            registry=registry,
        )

    def record_item_written(self, bytes_written: int) -> None:
        self._item_counter.inc()

    def record_flush(self, item_count: int, bytes_written: int, latency_seconds: float) -> None:
        self._flush_latency.observe(latency_seconds)
        self._flush_bytes.inc(bytes_written)

__all__ = ["PersistenceMetricsRecorder"]
