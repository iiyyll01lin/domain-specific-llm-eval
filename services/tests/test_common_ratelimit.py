"""Tests for services/common/ratelimit.py — TASK-092."""
import pytest
from unittest.mock import MagicMock

from services.common.ratelimit import RateLimiter


# ---------------------------------------------------------------------------
# RateLimiter logic
# ---------------------------------------------------------------------------


def test_first_request_allowed():
    limiter = RateLimiter(requests_per_minute=60)
    allowed, retry = limiter.is_allowed("1.2.3.4")
    assert allowed is True
    assert retry == 0


def test_burst_allows_multiple_requests():
    limiter = RateLimiter(requests_per_minute=60, burst=10)
    for _ in range(10):
        allowed, _ = limiter.is_allowed("1.2.3.4")
        assert allowed is True


def test_exceed_burst_rejected():
    # burst=1 → only 1 token initially, second request rejected immediately
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    limiter.is_allowed("1.2.3.4")  # consume the 1 token
    allowed, retry = limiter.is_allowed("1.2.3.4")
    assert allowed is False
    assert retry >= 1


def test_retry_after_positive_when_limited():
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    limiter.is_allowed("ip")  # consume
    _, retry = limiter.is_allowed("ip")
    assert retry > 0


def test_different_ips_independent():
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    limiter.is_allowed("ip1")  # exhaust ip1
    allowed, _ = limiter.is_allowed("ip2")
    assert allowed is True  # ip2 unaffected


def test_reset_single_ip():
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    limiter.is_allowed("ip1")
    limiter.reset("ip1")
    allowed, _ = limiter.is_allowed("ip1")
    assert allowed is True


def test_reset_all():
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    for ip in ["a", "b", "c"]:
        limiter.is_allowed(ip)
    limiter.reset()
    for ip in ["a", "b", "c"]:
        allowed, _ = limiter.is_allowed(ip)
        assert allowed is True


# ---------------------------------------------------------------------------
# rate_limit_dependency via FastAPI TestClient
# ---------------------------------------------------------------------------


def test_rate_limit_dependency_allows_request():
    """Integration: dependency does not raise when rate not exceeded."""
    import os
    os.environ["RATE_LIMIT_ENABLED"] = "1"
    os.environ["RATE_LIMIT_RPM"] = "1000"

    import importlib
    import services.common.ratelimit as rl_mod
    importlib.reload(rl_mod)

    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    from fastapi import Request

    app = FastAPI()

    @app.get("/test")
    async def endpoint(req: Request, _=Depends(rl_mod.rate_limit_dependency)):
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/test")
    assert resp.status_code == 200


def test_rate_limit_returns_429_when_exhausted():
    """Integration: dependency returns 429 when burst exceeded."""
    import os
    import importlib
    os.environ["RATE_LIMIT_ENABLED"] = "1"
    os.environ["RATE_LIMIT_RPM"] = "60"

    import services.common.ratelimit as rl_mod
    importlib.reload(rl_mod)

    # Replace the default limiter with a burst=1 variant
    rl_mod.default_limiter = RateLimiter(requests_per_minute=60, burst=1)

    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    from fastapi import Request

    app = FastAPI()

    @app.get("/limited")
    async def endpoint(req: Request, _=Depends(rl_mod.rate_limit_dependency)):
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    client.get("/limited")  # consume the one token
    resp = client.get("/limited")  # should be limited
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
