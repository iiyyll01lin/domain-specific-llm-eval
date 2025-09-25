import logging
import uuid
from typing import Iterable, Optional

import httpx

from services.common.config import settings
from services.common.events import EventPublisher, NullEventPublisher
from services.common.errors import ServiceError
from services.common.storage.object_store import ObjectStoreClient, compute_checksum
from services.ingestion.repository import IngestionJob, IngestionRepository

logger = logging.getLogger(__name__)


class KnowledgeManagementClient:
    """Simple HTTP client to retrieve document content from the KM API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client()
        self._timeout = timeout

    def iter_document_content(self, km_id: str, version: str) -> Iterable[bytes]:
        """Stream document bytes for the provided KM identifier and version."""
        path = f"/documents/{km_id}/versions/{version}/content"
        url = f"{self._base_url}{path}"
        with self._client.stream("GET", url, timeout=self._timeout) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk

    def close(self) -> None:  # pragma: no cover - convenience wrapper
        self._client.close()


class IngestionWorker:
    """Coordinates ingestion jobs: download, deduplicate, persist raw content."""

    def __init__(
        self,
        repository: IngestionRepository,
        km_client: Optional[KnowledgeManagementClient],
        object_store: ObjectStoreClient,
        *,
        bucket: Optional[str] = None,
        storage_prefix: str = "documents",
        event_publisher: Optional[EventPublisher] = None,
    ) -> None:
        self._repository = repository
        self._km_client = km_client
        self._object_store = object_store
        self._bucket = bucket or settings.object_store_bucket
        self._storage_prefix = storage_prefix.strip("/")
        self._events = event_publisher or NullEventPublisher()

    def process_job(self, job_id: str) -> IngestionJob:
        job = self._repository.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status not in {"queued", "retry"}:
            logger.info("Job %s already processed with status %s", job.job_id, job.status)
            return job

        existing = self._repository.get_document_by_km_version(job.km_id, job.version)
        if existing:
            logger.info("Job %s deduplicated by km/version -> %s", job.job_id, existing.document_id)
            self._repository.mark_job_completed(job.job_id, existing.document_id, deduplicated=True)
            updated = self._repository.get_job(job.job_id)
            if updated is None:  # pragma: no cover - defensive; repository always returns job after update
                raise RuntimeError("Failed to reload ingestion job after deduplication")
            return updated

        if not self._km_client:
            raise RuntimeError("KM client is required to process new documents")

        self._repository.mark_job_running(job.job_id)
        try:
            payload = self._download_document(job.km_id, job.version)
            checksum = compute_checksum(payload)

            existing_checksum = self._repository.get_document_by_checksum(checksum)
            if existing_checksum:
                logger.info(
                    "Job %s deduplicated by checksum -> %s", job.job_id, existing_checksum.document_id
                )
                self._repository.mark_job_completed(job.job_id, existing_checksum.document_id, deduplicated=True)
                updated = self._repository.get_job(job.job_id)
                if updated is None:  # pragma: no cover - defensive, same as above
                    raise RuntimeError("Failed to reload ingestion job after checksum deduplication")
                return updated

            document_id = uuid.uuid4().hex
            storage_key = self._build_storage_key(job.km_id, job.version, document_id)
            self._upload_document(storage_key, payload, checksum)
            document = self._repository.create_document(
                document_id=document_id,
                km_id=job.km_id,
                version=job.version,
                checksum=checksum,
                storage_key=storage_key,
                size_bytes=len(payload),
            )
            self._repository.mark_job_completed(job.job_id, document.document_id, deduplicated=False)
            self._events.document_ingested(
                document_id=document.document_id,
                checksum=document.checksum,
                byte_size=document.size_bytes,
                source_uri=f"km://{job.km_id}/{job.version}",
            )
            updated = self._repository.get_job(job.job_id)
            if updated is None:  # pragma: no cover - defensive
                raise RuntimeError("Failed to reload ingestion job after completion")
            return updated
        except ServiceError as exc:
            logger.exception("Job %s failed with service error", job.job_id)
            self._repository.mark_job_failed(job.job_id, exc.error_code, exc.message)
            raise
        except Exception as exc:  # pragma: no cover - defensive catch, covered by tests raising generic exception
            logger.exception("Job %s failed", job.job_id)
            self._repository.mark_job_failed(job.job_id, "ingestion_failed", str(exc))
            raise

    def _download_document(self, km_id: str, version: str) -> bytes:
        chunks = []
        for chunk in self._km_client.iter_document_content(km_id, version):  # type: ignore[union-attr]
            if not isinstance(chunk, (bytes, bytearray)):
                raise TypeError("KM client returned non-bytes chunk")
            chunks.append(bytes(chunk))
        if not chunks:
            raise ValueError("KM client returned empty payload")
        return b"".join(chunks)

    def _upload_document(self, storage_key: str, payload: bytes, checksum: str) -> None:
        if not self._bucket:
            raise ServiceError(
                error_code="object_store_bucket_missing",
                message="No object store bucket configured for ingestion",
            )
        self._object_store.upload_bytes(
            bucket=self._bucket,
            key=storage_key,
            payload=payload,
            expected_checksum=checksum,
        )

    def _build_storage_key(self, km_id: str, version: str, document_id: str) -> str:
        safe_km = km_id.replace("/", "_")
        safe_version = version.replace("/", "_")
        return f"{self._storage_prefix}/{safe_km}/{safe_version}/{document_id}.bin"


__all__ = ["KnowledgeManagementClient", "IngestionWorker", "IngestionJob"]
