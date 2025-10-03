from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

from services.eval.aggregation.aggregator import KPIAggregator, KPIAggregationResult
from services.eval.aggregation_metrics import AggregationMetricsRecorder
from services.eval.context_capture import CapturedEvaluationItem
from services.eval.kpi_writer import KPIWriter
from services.eval.persistence_metrics import PersistenceMetricsRecorder
from services.eval.stream_writer import EvaluationItemStreamWriter
from services.eval.backpressure import EvaluationItemQueueWorker


class EvaluationPersistencePipeline:
    """Coordinates persistence of evaluation items and KPI aggregation."""

    def __init__(
        self,
        run_id: str,
        output_dir: str | Path,
        *,
        flush_interval_seconds: float = 1.0,
        max_queue_size: int = 1024,
        persistence_metrics: Optional[PersistenceMetricsRecorder] = None,
        aggregation_metrics: Optional[AggregationMetricsRecorder] = None,
    ) -> None:
        self._run_id = run_id
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._items_path = self._output_dir / "evaluation_items.jsonl"
        self._kpis_path = self._output_dir / "kpis.json"
        self._persistence_metrics = persistence_metrics or PersistenceMetricsRecorder()
        self._stream_writer = EvaluationItemStreamWriter(
            self._items_path,
            flush_interval_seconds=flush_interval_seconds,
            manifest_path=self._output_dir / "evaluation_items.manifest.json",
            metrics=self._persistence_metrics,
        )
        self._queue_worker = EvaluationItemQueueWorker(
            self._stream_writer,
            max_queue_size=max_queue_size,
        )
        self._queue_lock = threading.Lock()
        self._closed = False
        self._metric_records: list[MutableMapping[str, float | None]] = []
        self._aggregator = KPIAggregator(metrics_recorder=aggregation_metrics)
        self._kpi_writer = KPIWriter(self._kpis_path)
        self._aggregation_result: Optional[KPIAggregationResult] = None

    @property
    def items_path(self) -> Path:
        return self._items_path

    @property
    def manifest_path(self) -> Path:
        return self._stream_writer.manifest_path

    @property
    def kpis_path(self) -> Path:
        return self._kpis_path

    @property
    def aggregation_result(self) -> Optional[KPIAggregationResult]:
        return self._aggregation_result

    def submit(self, item: CapturedEvaluationItem, metrics: Mapping[str, float | None]) -> None:
        with self._queue_lock:
            if self._closed:
                raise RuntimeError("Pipeline has been finalised")
            self._queue_worker.submit(item)
            self._metric_records.append(dict(metrics))

    def wait_until_drained(self, timeout: Optional[float] = None) -> bool:
        return self._queue_worker.join(timeout=timeout)

    def finalize(self) -> KPIAggregationResult:
        with self._queue_lock:
            if self._closed:
                if self._aggregation_result is None:
                    raise RuntimeError("Pipeline finalised without aggregation result")
                return self._aggregation_result
            self._queue_worker.stop()
            aggregation = self._aggregator.aggregate(self._metric_records)
            payload = {
                "run_id": self._run_id,
                "counts": dict(aggregation.counts),
                "metrics": {
                    name: {
                        "count": distribution.count,
                        "min": distribution.minimum,
                        "max": distribution.maximum,
                        "average": distribution.average,
                        "p50": distribution.p50,
                        "p95": distribution.p95,
                    }
                    for name, distribution in aggregation.metrics.items()
                },
            }
            self._kpi_writer.write(payload)
            self._aggregation_result = aggregation
            self._closed = True
            return aggregation

    def artifacts(self) -> Mapping[str, Path]:
        return {
            "items": self._items_path,
            "manifest": self.manifest_path,
            "kpis": self._kpis_path,
        }


__all__ = ["EvaluationPersistencePipeline"]
