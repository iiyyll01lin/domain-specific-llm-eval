import logging
from typing import Dict

import pytest

from services.common import config

_DEFAULT_ENV: Dict[str, str] = {
    "OBJECT_STORE_ENDPOINT": "http://localhost:9000",
    "OBJECT_STORE_REGION": "us-east-1",
    "OBJECT_STORE_ACCESS_KEY": "test-access-key",
    "OBJECT_STORE_SECRET_KEY": "test-secret-key",
    "OBJECT_STORE_BUCKET": "test-bucket",
    "OBJECT_STORE_USE_SSL": "false",
}


@pytest.mark.parametrize("missing_key", ["OBJECT_STORE_ENDPOINT", "OBJECT_STORE_ACCESS_KEY", "OBJECT_STORE_SECRET_KEY", "OBJECT_STORE_BUCKET"])
def test_missing_required_env_raises_runtime_error(monkeypatch, missing_key: str) -> None:
    for key in _DEFAULT_ENV.keys():
        if key == missing_key:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, _DEFAULT_ENV[key])

    with pytest.raises(RuntimeError):
        config.load_settings()

    # Restore environment to avoid polluting other tests.
    for key, value in _DEFAULT_ENV.items():
        monkeypatch.setenv(key, value)


def test_configuration_logged_once_with_redaction(caplog) -> None:
    config._reset_config_logging_for_tests()
    caplog.set_level(logging.INFO, logger="services.common.config")

    config.log_configuration(config.settings)
    config.log_configuration(config.settings)

    relevant_records = [record for record in caplog.records if record.getMessage() == "Loaded service configuration"]

    assert len(relevant_records) == 1
    snapshot = getattr(relevant_records[0], "config")
    assert snapshot["object_store_secret_key"] == "***redacted***"
    assert snapshot["object_store_access_key"] == "***redacted***"
    assert snapshot["object_store_endpoint"].startswith("http")
