"""Tests for services/common/auth.py — TASK-090."""
import os
import importlib

import pytest


def _reload_auth(env: dict):
    """Reload auth module with given env vars."""
    for k, v in env.items():
        os.environ[k] = v
    import services.common.auth as auth_mod
    importlib.reload(auth_mod)
    return auth_mod


def _clear_env():
    os.environ.pop("AUTH_TOKEN", None)
    os.environ.pop("AUTH_SCHEME", None)


class TestGetAuthHeader:
    def setup_method(self):
        _clear_env()

    def teardown_method(self):
        _clear_env()

    def test_no_token_returns_empty_dict(self):
        from services.common.auth import get_auth_header
        assert get_auth_header() == {}

    def test_token_set_returns_bearer_header(self):
        os.environ["AUTH_TOKEN"] = "mysecrettoken"
        mod = _reload_auth({})
        result = mod.get_auth_header()
        assert "Authorization" in result
        assert result["Authorization"].startswith("Bearer ")

    def test_token_value_in_header(self):
        os.environ["AUTH_TOKEN"] = "tok123"
        mod = _reload_auth({})
        assert "tok123" in mod.get_auth_header()["Authorization"]

    def test_custom_scheme(self):
        os.environ["AUTH_TOKEN"] = "apikey999"
        os.environ["AUTH_SCHEME"] = "ApiKey"
        mod = _reload_auth({})
        assert mod.get_auth_header()["Authorization"].startswith("ApiKey ")

    def test_is_auth_configured_false_when_no_token(self):
        from services.common import auth as auth_mod
        importlib.reload(auth_mod)
        assert auth_mod.is_auth_configured() is False

    def test_is_auth_configured_true_when_token(self):
        os.environ["AUTH_TOKEN"] = "t"
        mod = _reload_auth({})
        assert mod.is_auth_configured() is True

    def test_whitespace_only_token_returns_empty(self):
        os.environ["AUTH_TOKEN"] = "   "
        mod = _reload_auth({})
        assert mod.get_auth_header() == {}
