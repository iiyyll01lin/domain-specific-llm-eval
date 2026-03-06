"""TASK-092: Rate limit & backpressure hooks.

Provides a simple, dependency-injectable per-IP rate limiter for FastAPI
endpoints. Uses a token-bucket algorithm with a configurable window.

Usage (per endpoint)::

    from services.common.ratelimit import RateLimiter, rate_limit_dependency

    limiter = RateLimiter(requests_per_minute=60)
    app.state.rate_limiter = limiter

    @app.post("/jobs")
    async def create_job(request: Request, _=Depends(rate_limit_dependency)):
        ...

Environment variables
---------------------
RATE_LIMIT_RPM : int, default 60
    Requests per minute per source IP.
RATE_LIMIT_ENABLED : str, default "1"
    Set to "0" to disable rate limiting entirely (e.g. in tests).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import HTTPException, Request, status


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-process per-IP token-bucket rate limiter.

    Parameters
    ----------
    requests_per_minute : int
        Maximum sustained requests per 60-second window per client IP.
    burst : int, optional
        Maximum burst beyond the steady rate. Defaults to ``requests_per_minute``.
    """

    def __init__(self, requests_per_minute: int = 60, burst: int | None = None) -> None:
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst if burst is not None else requests_per_minute
        # ip → (tokens: float, last_check: float)
        self._buckets: Dict[str, Tuple[float, float]] = defaultdict(lambda: (float(self.burst), time.monotonic()))

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """Check whether *client_ip* is within its rate limit.

        Returns
        -------
        (allowed, retry_after_seconds)
        """
        tokens, last = self._buckets[client_ip]
        now = time.monotonic()
        elapsed = now - last
        tokens = min(self.burst, tokens + elapsed * self.rate)
        if tokens >= 1.0:
            self._buckets[client_ip] = (tokens - 1.0, now)
            return True, 0
        # Calculate seconds until next token is available
        retry_after = int((1.0 - tokens) / self.rate) + 1
        self._buckets[client_ip] = (tokens, now)
        return False, retry_after

    def reset(self, client_ip: str | None = None) -> None:
        """Reset bucket(s) — used in tests."""
        if client_ip is None:
            self._buckets.clear()
        else:
            self._buckets.pop(client_ip, None)


# ---------------------------------------------------------------------------
# Global default limiter (configured via env)
# ---------------------------------------------------------------------------

_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip() not in ("0", "false", "no")
_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))

default_limiter = RateLimiter(requests_per_minute=_RPM)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces the default rate limiter.

    Raises HTTP 429 with ``Retry-After`` header on limit exceeded.
    """
    if not _ENABLED:
        return
    ip = _client_ip(request)
    allowed, retry_after = default_limiter.is_allowed(ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )
