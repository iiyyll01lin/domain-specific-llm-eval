"""TASK-071: Event Envelope & Sequencing.

Defines the wire format for all WebSocket messages, plus helpers for:
- heartbeat production
- gap detection (client-side)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

MSG_TYPE_DATA = "data"
MSG_TYPE_HEARTBEAT = "heartbeat"
MSG_TYPE_ERROR = "error"
MSG_TYPE_WELCOME = "welcome"

VALID_TOPICS = frozenset(
    [
        "ingestion",
        "processing",
        "testset",
        "eval",
        "reporting",
        "kg",
    ]
)

HEARTBEAT_INTERVAL_SEC = 15
# Number of consecutive heartbeat misses before a downgrade trigger fires
HEARTBEAT_MISS_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Dataclass envelope
# ---------------------------------------------------------------------------


@dataclass
class Envelope:
    """Single WebSocket message sent to the client."""

    type: str  # MSG_TYPE_* constant
    seq: int  # monotonically increasing per-connection counter
    ts: float  # server Unix timestamp (seconds)
    topic: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "seq": self.seq,
            "ts": self.ts,
        }
        if self.topic is not None:
            d["topic"] = self.topic
        if self.payload:
            d["payload"] = self.payload
        return d


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_welcome(seq: int, subscribed_topics: List[str]) -> Envelope:
    return Envelope(
        type=MSG_TYPE_WELCOME,
        seq=seq,
        ts=time.time(),
        payload={"subscribed_topics": subscribed_topics},
    )


def make_heartbeat(seq: int) -> Envelope:
    return Envelope(
        type=MSG_TYPE_HEARTBEAT,
        seq=seq,
        ts=time.time(),
        payload={"interval_sec": HEARTBEAT_INTERVAL_SEC},
    )


def make_data(seq: int, topic: str, payload: Dict[str, Any]) -> Envelope:
    if topic not in VALID_TOPICS:
        raise ValueError(f"Unknown topic: {topic!r}")
    return Envelope(type=MSG_TYPE_DATA, seq=seq, ts=time.time(), topic=topic, payload=payload)


def make_error(seq: int, message: str) -> Envelope:
    return Envelope(type=MSG_TYPE_ERROR, seq=seq, ts=time.time(), payload={"message": message})


# ---------------------------------------------------------------------------
# Gap detection (server-side utility, also usable in tests)
# ---------------------------------------------------------------------------


def detect_gaps(received_seqs: List[int]) -> List[int]:
    """Return list of missing sequence numbers in the received list.

    ``received_seqs`` should be in the order the client received them.
    A gap is any integer *x* where ``prev_seq < x < current_seq`` and the
    gap is strictly > 1.

    Example::
        detect_gaps([1, 2, 5]) -> [3, 4]
    """
    if len(received_seqs) < 2:
        return []
    missing: List[int] = []
    for prev, curr in zip(received_seqs, received_seqs[1:]):
        if curr - prev > 1:
            missing.extend(range(prev + 1, curr))
    return missing


def should_resync(received_seqs: List[int]) -> bool:
    """Return True if any gap is detected (triggers REST resync)."""
    return bool(detect_gaps(received_seqs))
