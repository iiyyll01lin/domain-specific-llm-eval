import logging
import time
from typing import Any, Dict, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import get_trace_id

logger = logging.getLogger(__name__)

class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id", get_trace_id())
        request.state.trace_id = trace_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request.failed",
                extra={
                    "trace_id": trace_id,
                    "context": _build_log_context(
                        request=request,
                        status_code=_resolve_exception_status(exc),
                        duration_ms=duration_ms,
                    ),
                },
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["x-trace-id"] = trace_id
        logger.info(
            "request.completed",
            extra={
                "trace_id": trace_id,
                "context": _build_log_context(
                    request=request,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                ),
            },
        )
        return response


def _build_log_context(*, request: Request, status_code: int, duration_ms: float) -> Dict[str, Any]:
    client_host: Optional[str] = None
    if request.client:
        client_host = request.client.host
    return {
        "http_method": request.method,
        "http_path": request.url.path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "client_ip": client_host,
        "user_agent": request.headers.get("user-agent"),
    }


def _resolve_exception_status(exc: Exception) -> int:
    return getattr(exc, "status_code", getattr(exc, "http_status", 500))
