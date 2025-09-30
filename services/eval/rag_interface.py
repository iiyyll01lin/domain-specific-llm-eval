"""Core RAG adapter contract used by the evaluation service.

The evaluation microservice relies on adapter instances to talk to a
RAG-enabled system (HTTP endpoint, SDK wrapper, etc.). This module defines
lightweight request/response dataclasses along with the base adapter that
downstream implementations must extend. Keeping the contract in a dedicated
module allows us to unit test behaviours without requiring the eventual
network integration to exist yet (see TASK-031a).

Usage example::

    from services.eval.rag_interface import (
        RAGAdapter,
        RAGRequest,
        RAGResponse,
        RetrievedContext,
    )

    class MyAdapter(RAGAdapter):
        def invoke(self, request: RAGRequest) -> RAGResponse:
            payload = call_external_api(request.question)
            return RAGResponse(
                answer=payload["answer"],
                contexts=(
                    RetrievedContext(
                        document_id=item.get("id"),
                        text=item["text"],
                        score=item.get("score"),
                        metadata=item.get("metadata", {}),
                    )
                    for item in payload.get("contexts", [])
                ),
                latency_ms=payload["latency_ms"],
                raw=payload,
            )

Contract highlights:

* ``invoke`` must return a ``RAGResponse`` with the answer string and at least
  one context element (enforced later by the context capture wrapper).
* Implementations should populate ``raw`` with the original service payload for
  traceability/debugging. Sensitive values may be scrubbed but the structure
  should remain.
* Errors should be reported via the ``success`` flag and ``error_code`` rather
  than raising exceptions so that retry/backoff can make informed decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class RetrievedContext:
    """Represents a single piece of evidence returned by the RAG system."""

    text: str
    document_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation used by downstream writers."""

        return {
            "text": self.text,
            "document_id": self.document_id,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RAGRequest:
    """Encapsulates the information required to query a RAG target."""

    question: str
    run_id: str
    sample_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None


@dataclass(frozen=True)
class RAGResponse:
    """Normalised answer payload returned by a ``RAGAdapter`` implementation."""

    answer: str
    contexts: Tuple[RetrievedContext, ...]
    latency_ms: float
    success: bool = True
    error_code: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls,
        *,
        error_code: str,
        latency_ms: float = 0.0,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> "RAGResponse":
        """Helper constructor for adapters that surface a transport/service error."""

        return cls(
            answer="",
            contexts=(),
            latency_ms=latency_ms,
            success=False,
            error_code=error_code,
            raw=raw or {},
        )

    def ensure_contexts(self) -> "RAGResponse":
        """Return a copy making sure ``contexts`` is an immutable tuple.

        Adapters may provide iterable contexts; the evaluation pipeline expects a
        tuple for deterministic iteration and hashing. This helper converts the
        incoming payload while preserving immutability guarantees of the parent
        dataclass.
        """

        if isinstance(self.contexts, tuple):
            return self
        return RAGResponse(
            answer=self.answer,
            contexts=tuple(self.contexts),
            latency_ms=self.latency_ms,
            success=self.success,
            error_code=self.error_code,
            raw=self.raw,
        )


class RAGAdapter:
    """Base adapter responsible for invoking a RAG system.

    Implementations must override :meth:`invoke` and return a
    :class:`RAGResponse`. The default implementation raises
    :class:`NotImplementedError` so that unit tests enforcing the contract can
    detect missing overrides early.
    """

    def invoke(self, request: RAGRequest) -> RAGResponse:  # pragma: no cover - exercised in tests
        raise NotImplementedError("RAGAdapter.invoke must be implemented by subclasses")


class StaticResponseAdapter(RAGAdapter):
    """Simple in-memory adapter useful for tests and dry runs.

    The adapter returns a preconfigured answer/contexts pair allowing unit tests
    to exercise higher layers without performing network I/O. Although located
    alongside the contract, it is lightweight and side-effect free.
    """

    def __init__(
        self,
        *,
        answer: str,
        contexts: Iterable[RetrievedContext],
        latency_ms: float = 0.0,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._answer = answer
        self._contexts = tuple(contexts)
        self._latency_ms = latency_ms
        self._raw = raw or {}

    def invoke(self, request: RAGRequest) -> RAGResponse:
        return RAGResponse(
            answer=self._answer,
            contexts=self._contexts,
            latency_ms=self._latency_ms,
            success=True,
            error_code=None,
            raw=self._raw,
        )


__all__ = [
    "RAGAdapter",
    "RAGRequest",
    "RAGResponse",
    "RetrievedContext",
    "StaticResponseAdapter",
]
