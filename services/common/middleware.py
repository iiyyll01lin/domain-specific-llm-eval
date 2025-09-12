from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from .logging import get_trace_id

logger = logging.getLogger(__name__)

class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id", get_trace_id())
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers['x-trace-id'] = trace_id
        logger.info(f"request {request.method} {request.url.path}", extra={'trace_id': trace_id})
        return response
