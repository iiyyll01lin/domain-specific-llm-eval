from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from services.common.errors import ServiceError
from services.common.events import EventPublisher, NullEventPublisher
from services.ingestion.repository import DocumentRecord, IngestionRepository
from services.processing.repository import ProcessingRepository
from services.processing.stages import (
    ChunkBuilder,
    ChunkPersistence,
    ChunkPersistenceResult,
    ChunkingResult,
    EmbeddingBatchExecutor,
    EmbeddingExecutionResult,
    ExtractedDocument,
    TextExtractor,
)

logger = logging.getLogger(__name__)

_ALLOWED_STATUSES = {"queued", "retry"}


class ProcessingWorker:
    """Coordinates processing jobs from extraction through persistence and event emission."""

    def __init__(
        self,
        *,
        repository: ProcessingRepository,
        document_repository: IngestionRepository,
        text_extractor: TextExtractor,
        chunk_builder: ChunkBuilder,
        embedding_executor: EmbeddingBatchExecutor,
        chunk_persistence: ChunkPersistence,
        event_publisher: Optional[EventPublisher] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repository = repository
        self._document_repository = document_repository
        self._text_extractor = text_extractor
        self._chunk_builder = chunk_builder
        self._embedding_executor = embedding_executor
        self._chunk_persistence = chunk_persistence
        self._events = event_publisher or NullEventPublisher()
        self._clock = monotonic

    def process_job(self, job_id: str):  # type: ignore[no-untyped-def]
        job = self._repository.get_job(job_id)
        if job is None:
            raise ValueError(f"Processing job {job_id} not found")
        if job.status not in _ALLOWED_STATUSES:
            logger.info("Processing job %s already handled with status %s", job.job_id, job.status)
            return job

        document = self._document_repository.get_document(job.document_id)
        if document is None:
            error = ServiceError(
                error_code="document_not_found",
                message=f"Document {job.document_id} not found for processing",
                http_status=404,
            )
            self._repository.mark_job_failed(
                job.job_id,
                error_code=error.error_code,
                error_message=error.message,
            )
            raise error

        self._repository.mark_job_running(job.job_id)
        start_time = self._clock()
        try:
            extracted = self._extract_document(document)
            chunking_result = self._chunk_document(extracted)
            embedding_result = self._embed_chunks(chunking_result)
            persistence_result = self._persist_chunks(
                document=document,
                profile_hash=job.profile_hash,
                chunking_result=chunking_result,
                embedding_result=embedding_result,
            )
            self._repository.mark_job_completed(job.job_id)
            duration_ms = int(max(0.0, (self._clock() - start_time) * 1000))
            self._events.document_processed(
                document_id=document.document_id,
                profile_hash=job.profile_hash,
                chunk_count=len(chunking_result.chunks),
                embedding_count=len(embedding_result.embeddings),
                manifest_key=persistence_result.manifest_key,
                duration_ms=duration_ms,
            )
            updated = self._repository.get_job(job.job_id)
            if updated is None:  # pragma: no cover - defensive reload
                raise RuntimeError("Failed to reload processing job after completion")
            return updated
        except ServiceError as exc:
            logger.exception("Processing job %s failed with service error", job.job_id)
            self._repository.mark_job_failed(
                job.job_id,
                error_code=exc.error_code,
                error_message=exc.message,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive catch
            logger.exception("Processing job %s failed unexpectedly", job.job_id)
            self._repository.mark_job_failed(
                job.job_id,
                error_code="processing_failed",
                error_message=str(exc),
            )
            raise

    def _extract_document(self, document: DocumentRecord) -> ExtractedDocument:
        extracted = self._text_extractor.extract(document, expected_checksum=document.checksum)
        logger.info(
            "📄 document %s extracted",
            document.document_id,
            extra={"code": "PROCESS_EXTRACTED", "document_id": document.document_id},
        )
        return extracted

    def _chunk_document(self, extracted: ExtractedDocument) -> ChunkingResult:
        chunking_result = self._chunk_builder.build(document_text=extracted.text)
        if not chunking_result.chunks:
            raise ServiceError(
                error_code="chunk_generation_empty",
                message=f"Document {extracted.document_id} produced no chunks",
                http_status=422,
            )
        logger.info(
            "✂️ chunked document %s",
            extracted.document_id,
            extra={
                "code": "PROCESS_CHUNKED",
                "document_id": extracted.document_id,
                "chunk_count": len(chunking_result.chunks),
            },
        )
        return chunking_result

    def _embed_chunks(self, chunking_result: ChunkingResult) -> EmbeddingExecutionResult:
        embedding_result = self._embedding_executor.execute(chunking_result.chunks)
        logger.info(
            "🧠 embedded %s chunks",
            len(chunking_result.chunks),
            extra={"code": "PROCESS_EMBEDDED", "chunk_count": len(embedding_result.embeddings)},
        )
        return embedding_result

    def _persist_chunks(
        self,
        *,
        document: DocumentRecord,
        profile_hash: str,
        chunking_result: ChunkingResult,
        embedding_result: EmbeddingExecutionResult,
    ) -> ChunkPersistenceResult:
        metadata = {
            "km_id": document.km_id,
            "version": document.version,
            "checksum": document.checksum,
        }
        result = self._chunk_persistence.persist(
            document_id=document.document_id,
            profile_hash=profile_hash,
            chunks=chunking_result.chunks,
            embeddings=embedding_result,
            metadata=metadata,
        )
        logger.info(
            "💾 persisted chunks for %s",
            document.document_id,
            extra={
                "code": "PROCESS_PERSISTED",
                "document_id": document.document_id,
                "chunk_key": result.chunk_key,
                "manifest_key": result.manifest_key,
            },
        )
        return result


__all__ = ["ProcessingWorker"]
