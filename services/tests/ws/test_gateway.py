"""Tests for services/ws/gateway.py — TASK-070.

Uses httpx's AsyncClient + WebSocket testing support via starlette TestClient /
httpx AsyncClient (httpx >= 0.24 supports websocket).
"""
import json

import pytest
from fastapi.testclient import TestClient

from services.ws.gateway import app, manager


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_lists_valid_topics(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "valid_topics" in data
    assert "ingestion" in data["valid_topics"]


# ---------------------------------------------------------------------------
# WebSocket — successful handshake
# ---------------------------------------------------------------------------


def test_ws_welcome_envelope(client):
    with client.websocket_connect("/ui/events?topics=ingestion") as ws:
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "welcome"
    assert "ingestion" in msg["payload"]["subscribed_topics"]


def test_ws_welcome_has_seq(client):
    with client.websocket_connect("/ui/events?topics=eval") as ws:
        msg = json.loads(ws.receive_text())
    assert "seq" in msg
    assert isinstance(msg["seq"], int)


def test_ws_welcome_has_ts(client):
    with client.websocket_connect("/ui/events?topics=eval") as ws:
        msg = json.loads(ws.receive_text())
    assert "ts" in msg
    assert msg["ts"] > 0


def test_ws_no_topics_subscribes_all(client):
    with client.websocket_connect("/ui/events") as ws:
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "welcome"
    assert len(msg["payload"]["subscribed_topics"]) > 1


# ---------------------------------------------------------------------------
# WebSocket — unknown topic rejection
# ---------------------------------------------------------------------------


def test_ws_unknown_topic_error_envelope(client):
    with client.websocket_connect("/ui/events?topics=not-a-topic") as ws:
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "error"
    assert "Unknown topics" in msg["payload"]["message"]


# ---------------------------------------------------------------------------
# WebSocket — ping → heartbeat pong
# ---------------------------------------------------------------------------


def test_ws_ping_gets_heartbeat(client):
    with client.websocket_connect("/ui/events?topics=ingestion") as ws:
        _welcome = ws.receive_text()  # consume welcome
        ws.send_text(json.dumps({"type": "ping"}))
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "heartbeat"


# ---------------------------------------------------------------------------
# ConnectionManager unit tests
# ---------------------------------------------------------------------------


def test_manager_subscribe_and_unsubscribe():
    """Manager correctly tracks and removes subscriptions."""
    from unittest.mock import MagicMock

    ws = MagicMock()
    manager.subscribe(ws, ["ingestion"])
    assert ws in manager._subscriptions["ingestion"]
    manager.unsubscribe(ws)
    assert ws not in manager._subscriptions["ingestion"]


def test_manager_unsubscribe_nonexistent_no_error():
    from unittest.mock import MagicMock

    ws = MagicMock()
    # Should not raise even if ws was never added
    manager.unsubscribe(ws)
