from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import count
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class EventPublisher:
    """Lightweight in-process event publisher with pluggable transport."""

    def __init__(
        self,
        transport: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        sequence_start: int = 1,
    ) -> None:
        self._transport = transport or self._default_transport
        self._seq = count(sequence_start)

    def publish(self, event: str, payload: Dict[str, Any]) -> None:
        envelope = {
            "event": event,
            "seq": next(self._seq),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._transport(envelope)

    def document_ingested(
        self,
        *,
        document_id: str,
        checksum: str,
        byte_size: int,
        source_uri: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "document_id": document_id,
            "checksum": checksum,
            "byte_size": byte_size,
        }
        if source_uri:
            payload["source_uri"] = source_uri
        self.publish("document.ingested", payload)

    def document_processed(
        self,
        *,
        document_id: str,
        profile_hash: str,
        chunk_count: int,
        embedding_count: Optional[int] = None,
        manifest_key: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "document_id": document_id,
            "profile_hash": profile_hash,
            "chunk_count": chunk_count,
        }
        if embedding_count is not None:
            payload["embedding_count"] = embedding_count
        if manifest_key:
            payload["manifest_key"] = manifest_key
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        self.publish("document.processed", payload)

    def _default_transport(self, envelope: Dict[str, Any]) -> None:
        logger.info(
            "event emitted",
            extra={
                "event_name": envelope["event"],
                "event_seq": envelope["seq"],
            },
        )


class NullEventPublisher(EventPublisher):
    """Publisher that ignores all events."""

    def __init__(self) -> None:  # pragma: no cover - trivial wrapper
        super().__init__(transport=lambda _: None)


__all__ = ["EventPublisher", "NullEventPublisher"]
