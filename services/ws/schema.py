"""TASK-084: Event Schema JSON Validation for WS envelopes.

Provides:
- JSON Schema definitions for each envelope type
- ``validate_envelope(data)`` — validates a raw dict against the schema;
  raises ``EnvelopeValidationError`` on failure
- ``invalid_envelope_counter`` — in-process counter for monitoring
- Runtime validation toggle via env var ``WS_SCHEMA_VALIDATION`` (default "1")
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime toggle — set WS_SCHEMA_VALIDATION=0 to disable
# ---------------------------------------------------------------------------

_VALIDATION_ENABLED: bool = os.getenv("WS_SCHEMA_VALIDATION", "1").strip() not in ("0", "false", "no")

# ---------------------------------------------------------------------------
# In-process drop counter (incremented on validation failure)
# ---------------------------------------------------------------------------

invalid_envelope_counter: int = 0


def _increment_counter() -> None:
    global invalid_envelope_counter
    invalid_envelope_counter += 1


def reset_counter() -> None:
    """Reset counter — for testing."""
    global invalid_envelope_counter
    invalid_envelope_counter = 0


# ---------------------------------------------------------------------------
# JSON Schema — minimal subset sufficient for runtime validation
# ---------------------------------------------------------------------------

_BASE_REQUIRED = ["type", "seq", "ts"]

_BASE_PROPERTIES: Dict[str, Any] = {
    "type": {"type": "string", "enum": ["welcome", "heartbeat", "data", "error"]},
    "seq": {"type": "integer", "minimum": 0},
    "ts": {"type": "number", "exclusiveMinimum": 0},
    "topic": {"type": "string"},
    "payload": {"type": "object"},
}

ENVELOPE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WsEnvelope",
    "type": "object",
    "required": _BASE_REQUIRED,
    "properties": _BASE_PROPERTIES,
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class EnvelopeValidationError(ValueError):
    """Raised when an envelope dict fails schema validation."""


def _validate_schema(data: Dict[str, Any]) -> None:
    """Lightweight structural validation without jsonschema dependency.

    Raises ``EnvelopeValidationError`` on:
    - missing required fields
    - wrong field types
    - unknown additional properties
    - invalid enum value for ``type``
    """
    # Required fields
    for field in _BASE_REQUIRED:
        if field not in data:
            raise EnvelopeValidationError(f"Missing required field: {field!r}")

    # Type checks
    if not isinstance(data["type"], str):
        raise EnvelopeValidationError("Field 'type' must be a string")
    valid_types = {"welcome", "heartbeat", "data", "error"}
    if data["type"] not in valid_types:
        raise EnvelopeValidationError(f"Invalid type {data['type']!r}; expected one of {valid_types}")

    if not isinstance(data["seq"], int) or isinstance(data["seq"], bool):
        raise EnvelopeValidationError("Field 'seq' must be an integer")
    if data["seq"] < 0:
        raise EnvelopeValidationError("Field 'seq' must be >= 0")

    if not isinstance(data["ts"], (int, float)) or isinstance(data["ts"], bool):
        raise EnvelopeValidationError("Field 'ts' must be a number")
    if data["ts"] <= 0:
        raise EnvelopeValidationError("Field 'ts' must be > 0")

    # Optional field type checks
    if "topic" in data and not isinstance(data["topic"], str):
        raise EnvelopeValidationError("Field 'topic' must be a string")
    if "payload" in data and not isinstance(data["payload"], dict):
        raise EnvelopeValidationError("Field 'payload' must be an object")

    # No additional properties
    allowed = set(_BASE_PROPERTIES.keys())
    extra = set(data.keys()) - allowed
    if extra:
        raise EnvelopeValidationError(f"Additional properties not allowed: {extra!r}")


def validate_envelope(data: Dict[str, Any]) -> bool:
    """Validate *data* against the WsEnvelope schema.

    Returns
    -------
    bool
        ``True`` if valid (or validation is disabled).
        ``False`` if invalid (counter incremented, error logged).
    """
    if not _VALIDATION_ENABLED:
        return True

    try:
        _validate_schema(data)
        return True
    except EnvelopeValidationError as exc:
        _increment_counter()
        logger.warning("Invalid WS envelope dropped: %s | data=%r", exc, data)
        return False


def is_validation_enabled() -> bool:
    return _VALIDATION_ENABLED
