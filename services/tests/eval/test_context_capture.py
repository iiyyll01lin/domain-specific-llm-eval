from __future__ import annotations

from services.eval.context_capture import CapturedEvaluationItem, ContextCapture
from services.eval.rag_interface import RAGRequest, RAGResponse, RetrievedContext


def _request() -> RAGRequest:
    return RAGRequest(
        question="What is the capital of France?",
        run_id="run-123",
        sample_id="sample-456",
        metadata={"persona": "qa"},
        timeout_seconds=15,
        max_retries=2,
    )


def test_capture_preserves_contexts() -> None:
    capture = ContextCapture()
    response = RAGResponse(
        answer="Paris",
        contexts=(
            RetrievedContext(text="Paris is the capital.", document_id="doc-1", score=0.99),
            RetrievedContext(text="France capital info.", document_id="doc-2", score=0.87),
        ),
        latency_ms=45.6,
        success=True,
    )

    item = capture.capture(_request(), response)

    assert isinstance(item, CapturedEvaluationItem)
    assert item.answer == "Paris"
    assert len(item.contexts) == 2
    assert item.metadata["context_count"] == 2
    assert item.metadata["rag_latency_ms"] == 45.6
    assert item.metadata["persona"] == "qa"
    assert item.success is True
    assert item.error_code is None


def test_capture_injects_fallback_when_no_contexts() -> None:
    capture = ContextCapture()
    response = RAGResponse(
        answer="Paris",
        contexts=(),
        latency_ms=20.0,
        success=False,
        error_code="timeout",
    )

    item = capture.capture(_request(), response)

    assert len(item.contexts) == 1
    fallback = item.contexts[0]
    assert fallback.text.startswith("No supporting context")
    assert fallback.metadata["reason"] == "empty_context"
    assert fallback.metadata["run_id"] == "run-123"
    assert fallback.metadata["sample_id"] == "sample-456"
    assert item.metadata["context_count"] == 1
    assert item.success is False
    assert item.error_code == "timeout"