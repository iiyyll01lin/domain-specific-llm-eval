import os
import sys
from typing import Iterator, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.errors import ServiceError
from services.common.storage.object_store import compute_checksum
from services.ingestion.repository import IngestionRepository
from services.ingestion.worker import IngestionWorker


class FakeKMClient:
    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = chunks
        self.calls: List[Tuple[str, str]] = []

    def iter_document_content(self, km_id: str, version: str):
        self.calls.append((km_id, version))
        for chunk in self._chunks:
            yield chunk


class FakeObjectStore:
    def __init__(self) -> None:
        self.uploads: List[Tuple[Optional[str], str, bytes, Optional[str]]] = []
        self.fail_with: Optional[Exception] = None

    def upload_bytes(self, bucket, key, payload, expected_checksum=None):
        if self.fail_with:
            raise self.fail_with
        self.uploads.append((bucket, key, payload, expected_checksum))


@pytest.fixture
def repository(tmp_path) -> Iterator[IngestionRepository]:
    db_path = tmp_path / "ingestion.db"
    yield IngestionRepository(str(db_path))


def test_process_job_creates_document(repository: IngestionRepository):
    job = repository.create_job(km_id="KM-001", version="v1")
    km_client = FakeKMClient([b"hello", b" ", b"world"])
    object_store = FakeObjectStore()
    worker = IngestionWorker(
        repository=repository,
        km_client=km_client,
        object_store=object_store,
        bucket="test-bucket",
    )

    processed = worker.process_job(job.job_id)

    assert processed.status == "completed"
    assert processed.document_id is not None
    document = repository.get_document(processed.document_id)
    assert document is not None
    assert document.size_bytes == len(b"hello world")
    assert document.checksum == compute_checksum(b"hello world")
    assert object_store.uploads[0][1] == document.storage_key
    assert km_client.calls == [("KM-001", "v1")]


def test_process_job_reuses_existing_document(repository: IngestionRepository):
    km_client = FakeKMClient([b"payload"])
    object_store = FakeObjectStore()
    worker = IngestionWorker(
        repository=repository,
        km_client=km_client,
        object_store=object_store,
        bucket="test-bucket",
    )

    first_job = repository.create_job(km_id="KM-001", version="v1")
    processed_first = worker.process_job(first_job.job_id)
    assert processed_first.status == "completed"
    assert len(object_store.uploads) == 1

    second_job = repository.create_job(km_id="KM-001", version="v1")
    processed_second = worker.process_job(second_job.job_id)

    assert processed_second.status == "duplicate"
    assert processed_second.document_id == processed_first.document_id
    assert len(object_store.uploads) == 1
    assert km_client.calls == [("KM-001", "v1")]


def test_process_job_deduplicates_by_checksum(repository: IngestionRepository):
    km_client = FakeKMClient([b"same content"])
    object_store = FakeObjectStore()
    worker = IngestionWorker(
        repository=repository,
        km_client=km_client,
        object_store=object_store,
        bucket="test-bucket",
    )

    job_one = repository.create_job(km_id="KM-001", version="v1")
    processed_first = worker.process_job(job_one.job_id)
    assert processed_first.status == "completed"

    job_two = repository.create_job(km_id="KM-001", version="v2")
    processed_second = worker.process_job(job_two.job_id)

    assert processed_second.status == "duplicate"
    assert processed_second.document_id == processed_first.document_id
    assert len(object_store.uploads) == 1


def test_process_job_records_failure(repository: IngestionRepository):
    km_client = FakeKMClient([b"boom"])
    failure = ServiceError(error_code="object_store_unavailable", message="fail")
    object_store = FakeObjectStore()
    object_store.fail_with = failure
    worker = IngestionWorker(
        repository=repository,
        km_client=km_client,
        object_store=object_store,
        bucket="test-bucket",
    )

    job = repository.create_job(km_id="KM-ERR", version="v1")
    with pytest.raises(ServiceError):
        worker.process_job(job.job_id)

    stored_job = repository.get_job(job.job_id)
    assert stored_job is not None
    assert stored_job.status == "error"
    assert stored_job.error_code == failure.error_code
    assert stored_job.error_message == failure.message
