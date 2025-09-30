from __future__ import annotations

from typing import Protocol

from prometheus_client import Counter

_RUN_CREATED_COUNTER = Counter(
    "evaluation_run_created_total",
    "Total number of evaluation run submissions",
    labelnames=("result",),
)


class EvaluationRunMetricsRecorder(Protocol):
    def record(self, result: str) -> None:
        """Record evaluation run submission outcome."""


class PrometheusEvaluationRunMetrics(EvaluationRunMetricsRecorder):
    def __init__(self, counter: Counter = _RUN_CREATED_COUNTER) -> None:
        self._counter = counter

    def record(self, result: str) -> None:
        self._counter.labels(result=result).inc()


__all__ = ["EvaluationRunMetricsRecorder", "PrometheusEvaluationRunMetrics"]
