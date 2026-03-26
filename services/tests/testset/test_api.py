from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Generator, Tuple

import pytest
from fastapi.testclient import TestClient

from services.common.config import settings


class DummyMetrics:
    def __init__(self) -> None:
        self.results: list[str] = []

    def record(self, result: str) -> None:
        self.results.append(result)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[Tuple[TestClient, ModuleType, DummyMetrics], None, None]:
    db_path = tmp_path / "testset_jobs.db"
    monkeypatch.setattr(settings, "testset_db_path", str(db_path))

    module = importlib.import_module("services.testset.main")
    module = importlib.reload(module)

    module.get_repository.cache_clear()
    module.get_job_guard.cache_clear()
    module.get_metrics.cache_clear()

    metrics = DummyMetrics()

    app = module.app
    app.dependency_overrides[module.get_metrics] = lambda: metrics

    client = TestClient(app)
    try:
        yield client, module, metrics
    finally:
        client.close()
        app.dependency_overrides.clear()
        module.get_repository.cache_clear()
        module.get_job_guard.cache_clear()
        module.get_metrics.cache_clear()


def test_submit_testset_job_creates_job(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, module, metrics = client

    payload = {
        "method": "ragas",
        "max_total_samples": 25,
        "seed": 42,
        "selected_strategies": ["baseline"],
    }

    response = http_client.post("/testset-jobs", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["duplicate"] is False
    assert len(body["config_hash"]) == 12

    repository = module.get_repository()
    job = repository.get_job(body["job_id"])
    assert job is not None
    assert job.config_hash == body["config_hash"]
    assert job.method == "ragas"
    assert metrics.results == ["created"]


def test_duplicate_submission_returns_same_job_id(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, module, metrics = client

    payload = {"method": "configurable", "max_total_samples": 10}

    first = http_client.post("/testset-jobs", json=payload)
    second = http_client.post("/testset-jobs", json=payload)

    assert first.status_code == second.status_code == 202

    first_body = first.json()
    second_body = second.json()

    assert first_body["job_id"] == second_body["job_id"]
    assert first_body["duplicate"] is False
    assert second_body["duplicate"] is True
    assert metrics.results == ["created", "duplicate"]

    repository = module.get_repository()
    assert repository.count() == 1


def test_invalid_config_returns_error(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, _, metrics = client

    response = http_client.post("/testset-jobs", json={"method": "unsupported"})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "testset_config_invalid"
    assert metrics.results == []
