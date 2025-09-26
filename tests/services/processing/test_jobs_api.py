import os
import sys
import uuid
from typing import Iterator, Tuple

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.ingestion.repository import IngestionRepository
from services.processing.main import app, get_document_repository, get_repository
from services.processing.repository import ProcessingRepository


@pytest.fixture
def repositories(tmp_path) -> Iterator[Tuple[IngestionRepository, ProcessingRepository]]:
    ingestion_db = tmp_path / "ingestion.db"
    processing_db = tmp_path / "processing.db"
    ingestion_repo = IngestionRepository(str(ingestion_db))
    processing_repo = ProcessingRepository(str(processing_db))

    original_processing_override = app.dependency_overrides.get(get_repository)
    original_document_override = app.dependency_overrides.get(get_document_repository)

    app.dependency_overrides[get_repository] = lambda: processing_repo
    app.dependency_overrides[get_document_repository] = lambda: ingestion_repo

    try:
        yield ingestion_repo, processing_repo
    finally:
        if original_processing_override is not None:
            app.dependency_overrides[get_repository] = original_processing_override
        else:
            app.dependency_overrides.pop(get_repository, None)

        if original_document_override is not None:
            app.dependency_overrides[get_document_repository] = original_document_override
        else:
            app.dependency_overrides.pop(get_document_repository, None)


@pytest.fixture
def client(repositories) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(app)


def test_submit_processing_job_creates_record(
    client: TestClient,
    repositories: Tuple[IngestionRepository, ProcessingRepository],
) -> None:
    ingestion_repo, processing_repo = repositories
    document_id = uuid.uuid4().hex
    ingestion_repo.create_document(
        document_id=document_id,
        km_id="KM-001",
        version="v1",
        checksum="abc123",
        storage_key="s3://bucket/doc.bin",
        size_bytes=42,
    )

    payload = {"document_id": document_id, "profile_hash": "profile-xyz"}
    response = client.post("/process-jobs", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document_id
    assert body["profile_hash"] == payload["profile_hash"]
    assert body["status"] == "queued"

    stored_job = processing_repo.get_job(body["job_id"])
    assert stored_job is not None
    assert stored_job.document_id == document_id
    assert stored_job.profile_hash == payload["profile_hash"]
    assert processing_repo.count() == 1


def test_submit_processing_job_requires_existing_document(client: TestClient) -> None:
    payload = {"document_id": "missing-doc", "profile_hash": "profile-abc"}

    response = client.post("/process-jobs", json=payload)

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "document_not_found"


def test_submit_processing_job_validates_payload(client: TestClient) -> None:
    response = client.post("/process-jobs", json={"document_id": "", "profile_hash": ""})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "request_validation_failed"
