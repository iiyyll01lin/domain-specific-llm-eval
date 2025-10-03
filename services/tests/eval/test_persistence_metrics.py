from __future__ import annotations

from prometheus_client import CollectorRegistry

from services.eval.persistence_metrics import PersistenceMetricsRecorder


def test_metrics_recorder_updates_prometheus() -> None:
    registry = CollectorRegistry()
    recorder = PersistenceMetricsRecorder(registry=registry)
    recorder.record_item_written(bytes_written=100)
    recorder.record_item_written(bytes_written=200)
    recorder.record_flush(item_count=2, bytes_written=300, latency_seconds=0.05)

    item_count = registry.get_sample_value("eval_items_written_total")
    assert item_count == 2.0

    flush_bytes = registry.get_sample_value("eval_item_flush_bytes_total")
    assert flush_bytes == 300.0

    latency_bucket = registry.get_sample_value(
        "eval_item_flush_latency_seconds_bucket",
        labels={"le": "0.05"},
    )
    assert latency_bucket is not None
