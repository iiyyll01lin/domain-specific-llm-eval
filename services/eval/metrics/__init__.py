from __future__ import annotations

import logging
from typing import Optional, Protocol

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

_RUN_CREATED_COUNTER = Counter(
    "evaluation_run_created_total",
    "Total number of evaluation run submissions",
    labelnames=("result",),
)

_RAG_REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency distribution for outgoing RAG calls",
    labelnames=("outcome",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

_RAG_ATTEMPT_COUNTER = Counter(
    "rag_request_attempt_total",
    "Total number of RAG invocation attempts grouped by outcome",
    labelnames=("outcome",),
)


class EvaluationRunMetricsRecorder(Protocol):
    def record(self, result: str) -> None:
        """Record evaluation run submission outcome."""


class PrometheusEvaluationRunMetrics(EvaluationRunMetricsRecorder):
    def __init__(self, counter: Counter = _RUN_CREATED_COUNTER) -> None:
        self._counter = counter

    def record(self, result: str) -> None:
        self._counter.labels(result=result).inc()


class RAGRequestMetricsRecorder(Protocol):
    def record(
        self,
        *,
        outcome: str,
        latency_seconds: float,
        attempts: int,
        trace_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Record latency/attempts for a RAG invocation."""


class PrometheusRAGRequestMetrics(RAGRequestMetricsRecorder):
    def __init__(
        self,
        *,
        latency_histogram: Histogram = _RAG_REQUEST_LATENCY,
        attempts_counter: Counter = _RAG_ATTEMPT_COUNTER,
    ) -> None:
        self._latency = latency_histogram
        self._attempts = attempts_counter

    def record(
        self,
        *,
        outcome: str,
        latency_seconds: float,
        attempts: int,
        trace_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        safe_outcome = outcome if outcome in {"success", "failure"} else "unknown"
        self._latency.labels(outcome=safe_outcome).observe(max(latency_seconds, 0.0))
        self._attempts.labels(outcome=safe_outcome).inc(max(attempts, 1))

        logger.info(
            "rag.request_metrics",
            extra={
                "trace_id": trace_id,
                "context": {
                    "outcome": safe_outcome,
                    "latency_seconds": latency_seconds,
                    "attempts": attempts,
                    "error_code": error_code,
                },
            },
        )


__all__ = [
    "EvaluationRunMetricsRecorder",
    "PrometheusEvaluationRunMetrics",
    "RAGRequestMetricsRecorder",
    "PrometheusRAGRequestMetrics",
]
