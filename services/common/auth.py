"""TASK-090: Pluggable auth stub — no-op in dev, header injection in prod.

Usage:
    from services.common.auth import get_auth_header

    headers = get_auth_header()          # {}  when no token configured
    headers = get_auth_header()          # {"Authorization": "Bearer <token>"}

Environment variables
---------------------
AUTH_TOKEN : str, optional
    When set, ``get_auth_header()`` returns ``{"Authorization": "Bearer <token>"}``.
AUTH_SCHEME : str, default "Bearer"
    The auth scheme prefix (e.g. "Token", "ApiKey").
"""
from __future__ import annotations

import os
from typing import Dict, Optional


def get_auth_header() -> Dict[str, str]:
    """Return an ``Authorization`` header dict if ``AUTH_TOKEN`` is set.

    Returns an empty dict when running without a token (dev / no-op mode).
    """
    token = _get_token()
    if not token:
        return {}
    scheme = os.getenv("AUTH_SCHEME", "Bearer").strip()
    return {"Authorization": f"{scheme} {token}"}


def _get_token() -> Optional[str]:
    token = os.getenv("AUTH_TOKEN", "").strip()
    return token if token else None


def is_auth_configured() -> bool:
    """Return True when a token is present in the environment."""
    return _get_token() is not None
