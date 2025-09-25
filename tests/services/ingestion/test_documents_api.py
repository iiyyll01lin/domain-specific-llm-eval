import os
import sys
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.ingestion.main import app, get_repository
from services.ingestion.repository import IngestionRepository


@pytest.fixture
def temp_repo(tmp_path) -> Iterator[IngestionRepository]:
    db_path = tmp_path / "ingestion.db"
    repository = IngestionRepository(str(db_path))
    original_override = app.dependency_overrides.get(get_repository)
    app.dependency_overrides[get_repository] = lambda: repository
    try:
        yield repository
    finally:
        if original_override is not None:
            app.dependency_overrides[get_repository] = original_override
        else:
            app.dependency_overrides.pop(get_repository, None)


@pytest.fixture
def client(temp_repo: IngestionRepository) -> TestClient:
    return TestClient(app)


def test_submit_document_creates_job(client: TestClient, temp_repo: IngestionRepository):
    payload = {"km_id": "KM-001", "version": "v1"}

    response = client.post("/documents", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    job = temp_repo.get_job(job_id)
    assert job is not None
    assert job.km_id == payload["km_id"]
    assert job.version == payload["version"]
    assert temp_repo.count() == 1


def test_submit_document_validates_payload(client: TestClient):
    response = client.post("/documents", json={"km_id": "", "version": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "request_validation_failed"
