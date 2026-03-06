"""Tests for services/ws/schema.py — TASK-084."""
import pytest

from services.ws.schema import (
    EnvelopeValidationError,
    ENVELOPE_SCHEMA,
    invalid_envelope_counter,
    reset_counter,
    validate_envelope,
    _validate_schema,
)


def _valid_data_envelope():
    return {"type": "data", "seq": 1, "ts": 1700000000.0, "topic": "ingestion", "payload": {"count": 1}}


def _valid_heartbeat():
    return {"type": "heartbeat", "seq": 2, "ts": 1700000001.0}


# ---------------------------------------------------------------------------
# _validate_schema (internal)  -- tests pure validation logic
# ---------------------------------------------------------------------------


def test_valid_data_envelope_passes():
    _validate_schema(_valid_data_envelope())  # no exception


def test_valid_heartbeat_passes():
    _validate_schema(_valid_heartbeat())  # no exception


def test_missing_type_raises():
    d = {"seq": 1, "ts": 1.0}
    with pytest.raises(EnvelopeValidationError, match="type"):
        _validate_schema(d)


def test_missing_seq_raises():
    d = {"type": "heartbeat", "ts": 1.0}
    with pytest.raises(EnvelopeValidationError, match="seq"):
        _validate_schema(d)


def test_missing_ts_raises():
    d = {"type": "heartbeat", "seq": 1}
    with pytest.raises(EnvelopeValidationError, match="ts"):
        _validate_schema(d)


def test_invalid_type_value_raises():
    d = {"type": "unknown", "seq": 1, "ts": 1.0}
    with pytest.raises(EnvelopeValidationError, match="Invalid type"):
        _validate_schema(d)


def test_seq_negative_raises():
    d = {"type": "heartbeat", "seq": -1, "ts": 1.0}
    with pytest.raises(EnvelopeValidationError, match="seq"):
        _validate_schema(d)


def test_ts_zero_raises():
    d = {"type": "heartbeat", "seq": 1, "ts": 0}
    with pytest.raises(EnvelopeValidationError, match="ts"):
        _validate_schema(d)


def test_extra_field_raises():
    d = {**_valid_heartbeat(), "extra_field": "bad"}
    with pytest.raises(EnvelopeValidationError, match="Additional properties"):
        _validate_schema(d)


def test_payload_non_dict_raises():
    d = {"type": "data", "seq": 1, "ts": 1.0, "topic": "eval", "payload": "string"}
    with pytest.raises(EnvelopeValidationError, match="payload"):
        _validate_schema(d)


# ---------------------------------------------------------------------------
# validate_envelope (public)
# ---------------------------------------------------------------------------


def test_validate_envelope_valid_returns_true():
    reset_counter()
    assert validate_envelope(_valid_heartbeat()) is True


def test_validate_envelope_invalid_returns_false():
    reset_counter()
    result = validate_envelope({"type": "bad", "seq": 1, "ts": 1.0})
    assert result is False


def test_validate_envelope_increments_counter():
    reset_counter()
    validate_envelope({"type": "bad", "seq": 1, "ts": 1.0})
    from services.ws import schema as sch
    assert sch.invalid_envelope_counter == 1


def test_counter_increments_per_failure():
    reset_counter()
    validate_envelope({"type": "bad", "seq": 1, "ts": 1.0})
    validate_envelope({"seq": 1, "ts": 1.0})  # missing type
    from services.ws import schema as sch
    assert sch.invalid_envelope_counter == 2


# ---------------------------------------------------------------------------
# Schema dict structure
# ---------------------------------------------------------------------------


def test_schema_has_required_list():
    assert "required" in ENVELOPE_SCHEMA
    assert "type" in ENVELOPE_SCHEMA["required"]


def test_schema_has_properties():
    assert "properties" in ENVELOPE_SCHEMA
    assert "seq" in ENVELOPE_SCHEMA["properties"]
