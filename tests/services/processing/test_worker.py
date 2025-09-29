from __future__ import annotations

import os
import sys
from typing import Iterator, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.errors import ServiceError  # noqa: E402
from services.common.events import EventPublisher  # noqa: E402
from services.ingestion.repository import DocumentRecord, IngestionRepository  # noqa: E402
from services.processing.repository import ProcessingRepository  # noqa: E402
from services.processing.worker import ProcessingWorker  # noqa: E402
from services.processing.stages import (  # noqa: E402
    ChunkBuilder,
    ChunkCandidate,
    ChunkPersistenceItem,
    ChunkPersistenceResult,
    EmbeddingExecutionResult,
    ExtractedDocument,
)


class DummyTextExtractor:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract(self, document: DocumentRecord, *, expected_checksum: Optional[str] = None):  # type: ignore[no-untyped-def]
        return ExtractedDocument(
            document_id=document.document_id,
            checksum=document.checksum,
            text=self._text,
            byte_size=len(self._text.encode("utf-8")),
            mime_type="text/plain",
            metadata={"km_id": document.km_id, "version": document.version},
        )


class StubEmbeddingExecutor:
    def __init__(self) -> None:
        self.calls: List[List[ChunkCandidate]] = []

    def execute(self, chunks: List[ChunkCandidate]) -> EmbeddingExecutionResult:  # type: ignore[no-untyped-def]
        self.calls.append(chunks)
        embeddings = [[float(index)] for index, _chunk in enumerate(chunks)]
        sequence_indices = [chunk.sequence_index for chunk in chunks]
        return EmbeddingExecutionResult(embeddings=embeddings, sequence_indices=sequence_indices)


class RecordingChunkPersistence:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def persist(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return ChunkPersistenceResult(
            chunk_key="chunks/doc-1/profile/chunks.jsonl",
            embedding_key="chunks/doc-1/profile/embeddings.jsonl",
            manifest_key="chunks/doc-1/profile/manifest.json",
            chunk_checksum="abc",
            embedding_checksum="def",
            items=[ChunkPersistenceItem(sequence_index=0, token_count=3, chunk_checksum="abc", embedding_checksum="def")],
        )


class RecordingEventPublisher(EventPublisher):
    def __init__(self) -> None:
        super().__init__(transport=self._record)
        self.events: List[dict] = []

    def _record(self, envelope: dict) -> None:  # type: ignore[no-untyped-def]
        self.events.append(envelope)


@pytest.fixture
def repositories(tmp_path) -> Iterator[Tuple[IngestionRepository, ProcessingRepository]]:
    ingestion_db = tmp_path / "ingestion.db"
    processing_db = tmp_path / "processing.db"
    yield IngestionRepository(str(ingestion_db)), ProcessingRepository(str(processing_db))


def _make_document(repository: IngestionRepository) -> DocumentRecord:
    return repository.create_document(
        document_id="doc-1",
        km_id="KM-001",
        version="v1",
        checksum="checksum",
        storage_key="s3://bucket/doc.bin",
        size_bytes=12,
    )


def _make_job(repository: ProcessingRepository, document_id: str):  # type: ignore[no-untyped-def]
    return repository.create_job(document_id=document_id, profile_hash="profile-hash")


def test_processing_worker_processes_job_success(repositories) -> None:  # type: ignore[no-untyped-def]
    ingestion_repo, processing_repo = repositories
    document = _make_document(ingestion_repo)
    job = _make_job(processing_repo, document.document_id)

    text_extractor = DummyTextExtractor("hello world")
    chunk_builder = ChunkBuilder()
    embedding_executor = StubEmbeddingExecutor()
    chunk_persistence = RecordingChunkPersistence()
    events = RecordingEventPublisher()

    worker = ProcessingWorker(
        repository=processing_repo,
        document_repository=ingestion_repo,
        text_extractor=text_extractor,
        chunk_builder=chunk_builder,
        embedding_executor=embedding_executor,  # type: ignore[arg-type]
        chunk_persistence=chunk_persistence,  # type: ignore[arg-type]
        event_publisher=events,
    )

    updated_job = worker.process_job(job.job_id)

    assert updated_job.status == "completed"
    assert len(chunk_persistence.calls) == 1
    persistence_call = chunk_persistence.calls[0]
    assert persistence_call["document_id"] == document.document_id
    assert persistence_call["profile_hash"] == job.profile_hash
    assert persistence_call["metadata"]["km_id"] == document.km_id
    assert len(events.events) == 1
    envelope = events.events[0]
    assert envelope["event"] == "document.processed"
    payload = envelope["payload"]
    assert payload["chunk_count"] == 1
    assert payload["manifest_key"].endswith("manifest.json")


def test_processing_worker_handles_missing_document(repositories) -> None:  # type: ignore[no-untyped-def]
    ingestion_repo, processing_repo = repositories
    job = processing_repo.create_job(document_id="missing", profile_hash="profile")

    text_extractor = DummyTextExtractor("hello")
    worker = ProcessingWorker(
        repository=processing_repo,
        document_repository=ingestion_repo,
        text_extractor=text_extractor,
        chunk_builder=ChunkBuilder(),
        embedding_executor=StubEmbeddingExecutor(),  # type: ignore[arg-type]
        chunk_persistence=RecordingChunkPersistence(),  # type: ignore[arg-type]
    )

    with pytest.raises(ServiceError) as exc:
        worker.process_job(job.job_id)

    assert exc.value.error_code == "document_not_found"
    stored_job = processing_repo.get_job(job.job_id)
    assert stored_job is not None
    assert stored_job.status == "error"
    assert stored_job.error_code == "document_not_found"


def test_processing_worker_skips_non_queued_job(repositories) -> None:  # type: ignore[no-untyped-def]
    ingestion_repo, processing_repo = repositories
    document = _make_document(ingestion_repo)
    job = processing_repo.create_job(document_id=document.document_id, profile_hash="profile")
    processing_repo.mark_job_completed(job.job_id)

    worker = ProcessingWorker(
        repository=processing_repo,
        document_repository=ingestion_repo,
        text_extractor=DummyTextExtractor("text"),
        chunk_builder=ChunkBuilder(),
        embedding_executor=StubEmbeddingExecutor(),  # type: ignore[arg-type]
        chunk_persistence=RecordingChunkPersistence(),  # type: ignore[arg-type]
    )

    returned = worker.process_job(job.job_id)

    assert returned.status == "completed"