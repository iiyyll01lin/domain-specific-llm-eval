import io
import os
import sys
from typing import Any, Dict

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.errors import ChecksumMismatchError, ObjectStoreError
from services.common.storage.object_store import ObjectStoreClient, compute_checksum


class DummyClient:
    def __init__(self, responses: Dict[str, Any]):
        self._responses = responses
        self.calls = {"put_object": 0, "get_object": 0, "head_object": 0, "delete_object": 0}

    def put_object(self, **kwargs):
        self.calls["put_object"] += 1
        handler = self._responses.get("put_object")
        if callable(handler):
            return handler(**kwargs)
        return handler

    def get_object(self, **kwargs):
        self.calls["get_object"] += 1
        handler = self._responses.get("get_object")
        if callable(handler):
            return handler(**kwargs)
        return handler

    def head_object(self, **kwargs):
        self.calls["head_object"] += 1
        handler = self._responses.get("head_object")
        if callable(handler):
            return handler(**kwargs)
        return handler

    def delete_object(self, **kwargs):
        self.calls["delete_object"] += 1
        handler = self._responses.get("delete_object")
        if callable(handler):
            return handler(**kwargs)
        return handler


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("services.common.config.settings.object_store_bucket", "test-bucket")
    monkeypatch.setattr("services.common.config.settings.object_store_max_attempts", 3)
    monkeypatch.setattr("services.common.config.settings.object_store_backoff_seconds", 0)


def test_upload_bytes_success(mock_settings):
    payload = b"hello world"
    client = DummyClient({"put_object": {}})
    storage = ObjectStoreClient(client=client)

    checksum = storage.upload_bytes(bucket=None, key="foo.txt", payload=payload)

    assert checksum == compute_checksum(payload)
    assert client.calls["put_object"] == 1


def test_download_bytes_checksum_validation(mock_settings):
    payload = b"hello download"
    checksum = compute_checksum(payload)
    response = {
        "Body": io.BytesIO(payload),
        "Metadata": {"checksum": checksum},
    }
    client = DummyClient({"get_object": response})
    storage = ObjectStoreClient(client=client)

    content = storage.download_bytes(bucket=None, key="foo.txt")

    assert content == payload


def test_download_bytes_checksum_mismatch_raises(mock_settings):
    payload = b"bad data"
    response = {
        "Body": io.BytesIO(payload),
        "Metadata": {"checksum": "not-the-same"},
    }
    storage = ObjectStoreClient(client=DummyClient({"get_object": response}))

    with pytest.raises(ChecksumMismatchError):
        storage.download_bytes(bucket=None, key="foo.txt")


def test_retry_on_transient_failure_then_succeeds(mock_settings):
    payload = b"retry me"
    attempts = {"count": 0}

    def flaky_put_object(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ClientError(
                {"Error": {"Code": "RequestTimeout", "Message": "timeout"}},
                "PutObject",
            )
        return {}

    client = DummyClient({"put_object": flaky_put_object})
    storage = ObjectStoreClient(client=client, sleep_func=lambda _: None)

    checksum = storage.upload_bytes(bucket=None, key="foo.txt", payload=payload)

    assert checksum == compute_checksum(payload)
    assert attempts["count"] == 2


def test_object_exists_handles_not_found(mock_settings):
    error_response = {
        "Error": {"Code": "NoSuchKey", "Message": "test"}
    }

    def head_object(**kwargs):
        raise ClientError(error_response, "HeadObject")

    client = DummyClient({"head_object": head_object})
    storage = ObjectStoreClient(client=client, sleep_func=lambda _: None)

    assert storage.object_exists(bucket=None, key="missing.txt") is False


def test_object_exists_re_raises_other_errors(mock_settings):
    error_response = {
        "Error": {"Code": "InternalError", "Message": "boom"}
    }

    def head_object(**kwargs):
        raise ClientError(error_response, "HeadObject")

    client = DummyClient({"head_object": head_object})
    storage = ObjectStoreClient(client=client, sleep_func=lambda _: None)

    with pytest.raises(ObjectStoreError):
        storage.object_exists(bucket=None, key="other.txt")
