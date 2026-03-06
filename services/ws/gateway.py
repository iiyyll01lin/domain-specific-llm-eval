"""TASK-070: WebSocket Gateway Endpoint (Multiplex).

Single ``/ui/events`` WebSocket endpoint that:
- accepts a ``topics`` query-param (comma-separated, validated against VALID_TOPICS)
- sends a ``welcome`` envelope upon connection
- broadcasts a ``heartbeat`` envelope every HEARTBEAT_INTERVAL_SEC seconds
- streams data envelopes to subscribed topics via the ``broadcast()`` helper
- closes with 4000 on unknown topics, 4001 on internal error

Reconnect behaviour (client-side) is handled by ``useEventStream`` hook; the
gateway itself is stateless between reconnections.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError

from services.common.config import configure_service
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.ws.envelope import (
    HEARTBEAT_INTERVAL_SEC,
    VALID_TOPICS,
    make_data,
    make_error,
    make_heartbeat,
    make_welcome,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "ws-gateway"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


# ---------------------------------------------------------------------------
# Connection manager — tracks active websockets by subscribed topic
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections and topic subscriptions."""

    def __init__(self) -> None:
        # topic → set of active websockets
        self._subscriptions: dict[str, set[WebSocket]] = {t: set() for t in VALID_TOPICS}

    def subscribe(self, ws: WebSocket, topics: list[str]) -> None:
        for topic in topics:
            if topic in self._subscriptions:
                self._subscriptions[topic].add(ws)

    def unsubscribe(self, ws: WebSocket) -> None:
        for subs in self._subscriptions.values():
            subs.discard(ws)

    async def broadcast(self, topic: str, payload: dict, seq: int) -> None:
        """Send a data envelope to all subscribers of *topic*."""
        if topic not in self._subscriptions:
            return
        envelope = make_data(seq=seq, topic=topic, payload=payload)
        msg = json.dumps(envelope.to_dict())
        dead: list[WebSocket] = []
        for ws in list(self._subscriptions[topic]):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.unsubscribe(ws)


manager = ConnectionManager()

# Global sequence counter (per-process, not per-connection).
# For production use an atomic counter per connection; good-enough for tests.
_seq = 0


def _next_seq() -> int:
    global _seq
    _seq += 1
    return _seq


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/")
async def root() -> dict:
    return {
        "service": SERVICE_NAME,
        "ws_endpoint": "/ui/events?topics=<comma-separated>",
        "valid_topics": sorted(VALID_TOPICS),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ui/events")
async def ui_events(
    websocket: WebSocket,
    topics: str = Query(default="", description="Comma-separated list of topics to subscribe"),
) -> None:
    """Multiplex WebSocket endpoint.

    Query parameters
    ----------------
    topics : str
        Comma-separated topic names (e.g. ``topics=ingestion,eval``).
        An empty string subscribes to *all* valid topics.
    """
    await websocket.accept()

    # Parse and validate topics
    if topics.strip():
        requested: list[str] = [t.strip() for t in topics.split(",") if t.strip()]
    else:
        requested = list(VALID_TOPICS)

    unknown: list[str] = [t for t in requested if t not in VALID_TOPICS]
    if unknown:
        err_env = make_error(seq=_next_seq(), message=f"Unknown topics: {unknown!r}")
        await websocket.send_text(json.dumps(err_env.to_dict()))
        await websocket.close(code=4000)
        return

    manager.subscribe(websocket, requested)
    miss_count = 0

    try:
        # Send welcome envelope
        welcome = make_welcome(seq=_next_seq(), subscribed_topics=requested)
        await websocket.send_text(json.dumps(welcome.to_dict()))

        # Heartbeat loop — also listens for client messages
        while True:
            try:
                # Wait for a client message with a timeout = heartbeat interval
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=HEARTBEAT_INTERVAL_SEC,
                )
                # Client sent something; handle ping/control if needed
                try:
                    msg = json.loads(data)
                    msg_type = msg.get("type", "")
                    if msg_type == "ping":
                        pong = make_heartbeat(seq=_next_seq())
                        await websocket.send_text(json.dumps(pong.to_dict()))
                    # other client message types can be handled here
                except (json.JSONDecodeError, KeyError):
                    pass
                miss_count = 0  # received something → reset miss counter

            except asyncio.TimeoutError:
                # Heartbeat interval elapsed — send heartbeat
                hb = make_heartbeat(seq=_next_seq())
                try:
                    await websocket.send_text(json.dumps(hb.to_dict()))
                    miss_count = 0
                except Exception:  # noqa: BLE001
                    miss_count += 1
                    logger.warning("Heartbeat send failed (miss %d)", miss_count)
                    if miss_count >= 2:
                        # Two consecutive misses → close; client will downgrade
                        logger.info("Closing WS after %d consecutive heartbeat misses", miss_count)
                        break

    except WebSocketDisconnect:
        logger.debug("Client disconnected from /ui/events")
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in /ui/events handler")
    finally:
        manager.unsubscribe(websocket)


# ---------------------------------------------------------------------------
# Internal broadcast helper — exposed for service-to-service calls
# ---------------------------------------------------------------------------


async def broadcast(topic: str, payload: dict) -> None:
    """Broadcast a data event to all subscribers of *topic*."""
    await manager.broadcast(topic=topic, payload=payload, seq=_next_seq())
