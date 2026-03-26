from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Generator, Tuple

import pytest
from fastapi.testclient import TestClient

from services.common.config import settings
from services.testset.config_normalizer import compute_config_hash


class DummyMetrics:
    def __init__(self) -> None:
        self.results: list[str] = []

    def record(self, result: str) -> None:
        self.results.append(result)


def _seed_completed_testset(module: ModuleType, *, method: str = "ragas") -> str:
    repository = module.get_testset_repository()
    config = {"method": method, "max_total_samples": 10}
    config_hash = compute_config_hash(config)
    job = repository.create_job(config_hash=config_hash, config=config)
    repository.mark_completed(
        job.job_id,
        sample_count=5,
        persona_count=2,
        scenario_count=2,
        seed=42,
        artifact_prefix="tests/seed",
        artifact_paths={},
        metadata={},
    )
    return job.job_id


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Generator[Tuple[TestClient, ModuleType, DummyMetrics], None, None]:
    eval_db = tmp_path / "eval_runs.db"
    testset_db = tmp_path / "testset_jobs.db"
    monkeypatch.setattr(settings, "eval_db_path", str(eval_db))
    monkeypatch.setattr(settings, "testset_db_path", str(testset_db))

    module = importlib.import_module("services.eval.main")
    module = importlib.reload(module)

    module.get_repository.cache_clear()
    module.get_testset_repository.cache_clear()
    module.get_guard.cache_clear()
    module.get_metrics.cache_clear()
    module.get_validator.cache_clear()

    metrics = DummyMetrics()

    app = module.app
    app.dependency_overrides[module.get_metrics] = lambda: metrics

    http_client = TestClient(app)
    try:
        yield http_client, module, metrics
    finally:
        http_client.close()
        app.dependency_overrides.clear()
        module.get_repository.cache_clear()
        module.get_testset_repository.cache_clear()
        module.get_guard.cache_clear()
        module.get_metrics.cache_clear()
        module.get_validator.cache_clear()


def test_submit_eval_run_creates_run(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, module, metrics = client
    testset_id = _seed_completed_testset(module)

    payload = {
        "testset_id": testset_id,
        "profile": "baseline",
        "rag_endpoint": "http://localhost:8080/query",
    }

    response = http_client.post("/eval-runs", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["duplicate"] is False
    assert len(body["run_id"]) == 32

    repository = module.get_repository()
    run = repository.get_run(body["run_id"])
    assert run is not None
    assert run.testset_id == testset_id
    assert run.profile == "baseline"
    assert metrics.results == ["created"]


def test_duplicate_submission_returns_same_run(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, module, metrics = client
    testset_id = _seed_completed_testset(module, method="configurable")

    payload = {"testset_id": testset_id, "profile": "baseline"}

    first = http_client.post("/eval-runs", json=payload)
    second = http_client.post("/eval-runs", json=payload)

    assert first.status_code == second.status_code == 202
    first_body = first.json()
    second_body = second.json()

    assert first_body["run_id"] == second_body["run_id"]
    assert first_body["duplicate"] is False
    assert second_body["duplicate"] is True
    assert metrics.results == ["created", "duplicate"]

    repository = module.get_repository()
    assert repository.count() == 1


def test_missing_testset_returns_error(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, _, metrics = client

    payload = {
        "testset_id": "550e8400e29b41d4a716446655440000",
        "profile": "baseline",
    }

    response = http_client.post("/eval-runs", json=payload)

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "testset_not_found"
    assert metrics.results == []


def test_invalid_profile_returns_error(client: Tuple[TestClient, ModuleType, DummyMetrics]) -> None:
    http_client, module, metrics = client
    testset_id = _seed_completed_testset(module)

    payload = {
        "testset_id": testset_id,
        "profile": "invalid_profile",
    }

    response = http_client.post("/eval-runs", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_profile"
    assert "available profiles" in body["message"].lower()
    assert metrics.results == []
