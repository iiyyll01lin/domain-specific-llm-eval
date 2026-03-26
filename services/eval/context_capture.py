"""Utilities for normalising contexts returned by a RAG adapter.

Task reference: TASK-031b. The evaluation runner stores every retrieved
context to guarantee traceability. Some upstream systems, however, may omit
contexts or return ``None`` when they fail to retrieve supporting documents.
The :class:`ContextCapture` wrapper centralises the normalisation logic so that
later pipeline stages can assume ``contexts`` is a non-empty tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from services.eval.rag_interface import (
    RAGRequest,
    RAGResponse,
    RetrievedContext,
)

_FALLBACK_TEMPLATE = "No supporting context returned by RAG provider"


@dataclass(frozen=True)
class CapturedEvaluationItem:
    """Payload ready to be persisted after a RAG invocation."""

    run_id: str
    sample_id: str
    question: str
    answer: str
    contexts: Tuple[RetrievedContext, ...]
    success: bool
    error_code: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    raw: Mapping[str, object] = field(default_factory=dict)

    def with_contexts(self, contexts: Tuple[RetrievedContext, ...]) -> "CapturedEvaluationItem":
        return CapturedEvaluationItem(
            run_id=self.run_id,
            sample_id=self.sample_id,
            question=self.question,
            answer=self.answer,
            contexts=contexts,
            success=self.success,
            error_code=self.error_code,
            metadata=self.metadata,
            raw=self.raw,
        )


class ContextCapture:
    """Guarantees downstream consumers always receive at least one context."""

    def __init__(self, *, fallback_text: str = _FALLBACK_TEMPLATE) -> None:
        self._fallback_text = fallback_text

    def capture(self, request: RAGRequest, response: RAGResponse) -> CapturedEvaluationItem:
        contexts = tuple(response.contexts)
        if not contexts:
            contexts = (
                RetrievedContext(
                    text=self._fallback_text,
                    document_id=None,
                    score=None,
                    metadata={
                        "run_id": request.run_id,
                        "sample_id": request.sample_id,
                        "reason": "empty_context",
                    },
                ),
            )

        enriched_metadata = dict(request.metadata)
        enriched_metadata.update(
            {
                "rag_latency_ms": response.latency_ms,
                "context_count": len(contexts),
            }
        )

        return CapturedEvaluationItem(
            run_id=request.run_id,
            sample_id=request.sample_id,
            question=request.question,
            answer=response.answer,
            contexts=contexts,
            success=response.success,
            error_code=response.error_code,
            metadata=enriched_metadata,
            raw=dict(response.raw),
        )


__all__ = ["CapturedEvaluationItem", "ContextCapture"]
