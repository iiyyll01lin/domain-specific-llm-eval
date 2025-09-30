from __future__ import annotations

from typing import Dict, List, Optional
from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers.testset_schema import TestsetSample

from services.eval.context_capture import ContextCapture
from services.eval.metrics import RAGRequestMetricsRecorder
from services.eval.rag_interface import (
    RAGAdapter,
    RAGRequest,
    RAGResponse,
    RetrievedContext,
    StaticResponseAdapter,
)
from services.eval.retry_policy import RAGInvocationError, RetryPolicy
from services.eval.runner import EvaluationRunner


def _retry_factory(max_attempts: int) -> RetryPolicy:
    return RetryPolicy(max_attempts=max_attempts, base_delay_seconds=0.0, sleep_fn=lambda _: None)


def _sample(*, question: str = "What is the policy?", answer: str = "The policy mandates annual reviews.") -> TestsetSample:
    return TestsetSample(
        eval_sample=SingleTurnSample(
            user_input=question,
            reference=answer,
            reference_contexts=["Compliance policies require annual review of processes."],
            rubrics={"strategy": "baseline"},
        ),
        synthesizer_name="ragas",
    )


class RecordingMetrics(RAGRequestMetricsRecorder):
    def __init__(self) -> None:
        self.records: List[Dict[str, object]] = []

    def record(
        self,
        *,
        outcome: str,
        latency_seconds: float,
        attempts: int,
        trace_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        self.records.append(
            {
                "outcome": outcome,
                "latency_seconds": latency_seconds,
                "attempts": attempts,
                "trace_id": trace_id,
                "error_code": error_code,
            }
        )


def test_runner_captures_contexts() -> None:
    sample = _sample()
    adapter = StaticResponseAdapter(
        answer="Annual reviews are mandated.",
        contexts=(
            RetrievedContext(
                text="Compliance policies require annual reviews of processes.",
                document_id="doc-1",
                score=0.91,
                metadata={"source": "chunk:1"},
            ),
        ),
        latency_ms=25.0,
    )
    metrics = RecordingMetrics()
    runner = EvaluationRunner(
        adapter=adapter,
        metrics_recorder=metrics,
        retry_policy_factory=_retry_factory,
    )

    items = list(
        runner.run(
            run_id="run-123",
            samples=[sample],
            timeout_seconds=30.0,
            max_retries=2,
        )
    )

    assert len(items) == 1
    item = items[0]
    assert item.success is True
    assert item.contexts[0].text.startswith("Compliance policies")
    assert item.metadata["reference_answer"] == sample.eval_sample.reference
    assert item.metadata["rag_attempts"] == 1
    assert metrics.records[0]["outcome"] == "success"
    assert metrics.records[0]["attempts"] == 1


def test_runner_injects_fallback_context_when_missing() -> None:
    sample = _sample()
    adapter = StaticResponseAdapter(
        answer="",
        contexts=tuple(),
        latency_ms=12.0,
    )
    metrics = RecordingMetrics()
    runner = EvaluationRunner(
        adapter=adapter,
        metrics_recorder=metrics,
        retry_policy_factory=_retry_factory,
        context_capture=ContextCapture(),
    )

    items = list(
        runner.run(
            run_id="run-456",
            samples=[sample],
            timeout_seconds=15.0,
            max_retries=1,
        )
    )

    assert len(items) == 1
    item = items[0]
    assert item.success is True
    assert len(item.contexts) == 1
    assert item.contexts[0].metadata["reason"] == "empty_context"
    assert item.metadata["rag_attempts"] == 1
    assert metrics.records[0]["outcome"] == "success"


class FlakyAdapter(RAGAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: RAGRequest) -> RAGResponse:
        self.calls += 1
        if self.calls == 1:
            raise RAGInvocationError("throttled", status_code=429, error_code="throttle")
        return RAGResponse(
            answer="Final answer",
            contexts=(
                RetrievedContext(
                    text=f"Context for {request.sample_id}",
                    document_id="doc-77",
                    score=0.8,
                    metadata={"source": "retry"},
                ),
            ),
            latency_ms=18.5,
        )


class FailingAdapter(RAGAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: RAGRequest) -> RAGResponse:
        self.calls += 1
        raise RAGInvocationError(
            f"upstream failure #{self.calls}",
            status_code=503,
            error_code="service_unavailable",
        )


def test_runner_retries_on_transient_failure() -> None:
    sample = _sample()
    adapter = FlakyAdapter()
    metrics = RecordingMetrics()
    runner = EvaluationRunner(
        adapter=adapter,
        metrics_recorder=metrics,
        retry_policy_factory=_retry_factory,
    )

    items = list(
        runner.run(
            run_id="run-789",
            samples=[sample],
            timeout_seconds=20.0,
            max_retries=3,
        )
    )

    assert adapter.calls == 2
    assert len(items) == 1
    item = items[0]
    assert item.success is True
    assert item.metadata["rag_attempts"] == 2
    assert metrics.records[0]["attempts"] == 2
    assert metrics.records[0]["outcome"] == "success"


def test_runner_returns_failure_item_after_max_retries() -> None:
    sample = _sample()
    adapter = FailingAdapter()
    metrics = RecordingMetrics()
    runner = EvaluationRunner(
        adapter=adapter,
        metrics_recorder=metrics,
        retry_policy_factory=_retry_factory,
    )

    items = list(
        runner.run(
            run_id="run-999",
            samples=[sample],
            timeout_seconds=10.0,
            max_retries=2,
        )
    )

    assert adapter.calls == 2
    assert len(items) == 1
    item = items[0]
    assert item.success is False
    assert item.metadata["rag_outcome"] == "failure"
    assert item.metadata["rag_attempts"] == 2
    assert item.metadata["rag_error_code"] == "service_unavailable"
    assert item.contexts[0].metadata["reason"] == "empty_context"
    assert metrics.records[0]["outcome"] == "failure"
    assert metrics.records[0]["attempts"] == 2
