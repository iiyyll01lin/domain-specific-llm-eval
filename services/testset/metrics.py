from __future__ import annotations

from typing import Protocol

from prometheus_client import Counter

_JOB_CREATED_COUNTER = Counter(
    "testset_job_created_total",
    "Total number of testset job submissions",
    labelnames=("result",),
)


class TestsetJobMetricsRecorder(Protocol):
    def record(self, result: str) -> None:
        """Record a job submission with the given *result* label."""


class PrometheusTestsetJobMetrics(TestsetJobMetricsRecorder):
    def __init__(self, counter: Counter = _JOB_CREATED_COUNTER) -> None:
        self._counter = counter

    def record(self, result: str) -> None:
        self._counter.labels(result=result).inc()


__all__ = [
    "PrometheusTestsetJobMetrics",
    "TestsetJobMetricsRecorder",
]
