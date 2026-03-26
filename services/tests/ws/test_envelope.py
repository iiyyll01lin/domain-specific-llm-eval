"""Tests for services/ws/envelope.py — TASK-071."""
import pytest

from services.ws.envelope import (
    HEARTBEAT_INTERVAL_SEC,
    VALID_TOPICS,
    detect_gaps,
    make_data,
    make_error,
    make_heartbeat,
    make_welcome,
    should_resync,
)


# ---------------------------------------------------------------------------
# make_welcome
# ---------------------------------------------------------------------------


def test_make_welcome_type():
    env = make_welcome(seq=1, subscribed_topics=["ingestion"])
    assert env.type == "welcome"


def test_make_welcome_seq():
    env = make_welcome(seq=7, subscribed_topics=["eval"])
    assert env.seq == 7


def test_make_welcome_payload_contains_topics():
    env = make_welcome(seq=1, subscribed_topics=["ingestion", "eval"])
    assert "ingestion" in env.payload["subscribed_topics"]
    assert "eval" in env.payload["subscribed_topics"]


# ---------------------------------------------------------------------------
# make_heartbeat
# ---------------------------------------------------------------------------


def test_make_heartbeat_type():
    env = make_heartbeat(seq=2)
    assert env.type == "heartbeat"


def test_make_heartbeat_interval_in_payload():
    env = make_heartbeat(seq=3)
    assert env.payload["interval_sec"] == HEARTBEAT_INTERVAL_SEC


def test_make_heartbeat_ts_positive():
    env = make_heartbeat(seq=1)
    assert env.ts > 0


# ---------------------------------------------------------------------------
# make_data
# ---------------------------------------------------------------------------


def test_make_data_type():
    env = make_data(seq=5, topic="ingestion", payload={"count": 1})
    assert env.type == "data"


def test_make_data_topic_set():
    env = make_data(seq=5, topic="eval", payload={})
    assert env.topic == "eval"


def test_make_data_unknown_topic_raises():
    with pytest.raises(ValueError, match="Unknown topic"):
        make_data(seq=1, topic="not-a-topic", payload={})


# ---------------------------------------------------------------------------
# make_error
# ---------------------------------------------------------------------------


def test_make_error_type():
    env = make_error(seq=1, message="oops")
    assert env.type == "error"


def test_make_error_message_in_payload():
    env = make_error(seq=1, message="bad topic")
    assert env.payload["message"] == "bad topic"


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_has_type_seq_ts():
    env = make_heartbeat(seq=99)
    d = env.to_dict()
    assert "type" in d
    assert "seq" in d
    assert "ts" in d


def test_to_dict_omits_topic_when_none():
    env = make_heartbeat(seq=1)
    d = env.to_dict()
    assert "topic" not in d


def test_to_dict_includes_topic_for_data():
    env = make_data(seq=1, topic="kg", payload={"x": 1})
    d = env.to_dict()
    assert d["topic"] == "kg"


# ---------------------------------------------------------------------------
# detect_gaps / should_resync
# ---------------------------------------------------------------------------


def test_detect_gaps_no_gap():
    assert detect_gaps([1, 2, 3, 4]) == []


def test_detect_gaps_single_gap():
    assert detect_gaps([1, 2, 5]) == [3, 4]


def test_detect_gaps_multiple_gaps():
    result = detect_gaps([1, 3, 7])
    assert 2 in result
    assert 4 in result
    assert 5 in result
    assert 6 in result


def test_detect_gaps_empty():
    assert detect_gaps([]) == []


def test_detect_gaps_single_element():
    assert detect_gaps([5]) == []


def test_should_resync_true_on_gap():
    assert should_resync([1, 3]) is True


def test_should_resync_false_no_gap():
    assert should_resync([1, 2, 3]) is False


# ---------------------------------------------------------------------------
# VALID_TOPICS sanity
# ---------------------------------------------------------------------------


def test_valid_topics_non_empty():
    assert len(VALID_TOPICS) > 0


def test_valid_topics_include_ingestion_eval():
    assert "ingestion" in VALID_TOPICS
    assert "eval" in VALID_TOPICS
