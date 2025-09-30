from __future__ import annotations

import pytest

from services.eval.rag_interface import (
    RAGAdapter,
    RAGRequest,
    RAGResponse,
    RetrievedContext,
    StaticResponseAdapter,
)


def test_base_adapter_invoke_not_implemented() -> None:
    adapter = RAGAdapter()
    request = RAGRequest(question="What is RAG?", run_id="run-1", sample_id="sample-1")

    with pytest.raises(NotImplementedError):
        adapter.invoke(request)


def test_static_adapter_returns_configured_response() -> None:
    adapter = StaticResponseAdapter(
        answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=[
            RetrievedContext(text="Definition snippet", document_id="doc-1", score=0.91),
            RetrievedContext(text="Another snippet", document_id="doc-2", score=0.73),
        ],
        latency_ms=123.4,
        raw={"answer": "...", "contexts": []},
    )

    response = adapter.invoke(RAGRequest(question="Explain RAG", run_id="r", sample_id="s"))

    assert response.answer.startswith("RAG stands for")
    assert len(response.contexts) == 2
    assert response.latency_ms == pytest.approx(123.4)
    assert response.success is True
    assert response.raw["answer"] == "..."


def test_response_ensure_contexts_normalises_iterable() -> None:
    response = RAGResponse(
        answer="test",
        contexts=(RetrievedContext(text="a"),),
        latency_ms=10.0,
    )
    assert response.ensure_contexts() is response

    response_with_list = RAGResponse(
        answer="test",
        contexts=[RetrievedContext(text="a"), RetrievedContext(text="b")],
        latency_ms=10.0,
    )
    normalised = response_with_list.ensure_contexts()
    assert normalised is not response_with_list
    assert isinstance(normalised.contexts, tuple)
    assert [ctx.text for ctx in normalised.contexts] == ["a", "b"]
