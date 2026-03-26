from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from services.common.errors import ServiceError
from services.common.storage.object_store import ObjectStoreClient, compute_checksum
from services.processing.stages.chunk_rules import ChunkCandidate
from services.processing.stages.embed_executor import EmbeddingExecutionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkPersistenceConfig:
    """Configuration controlling chunk persistence behaviour."""

    bucket: Optional[str] = None
    storage_prefix: str = "chunks"
    chunks_filename: str = "chunks.jsonl"
    embeddings_filename: str = "embeddings.jsonl"
    manifest_filename: str = "manifest.json"

    def __post_init__(self) -> None:
        prefix = (self.storage_prefix or "").strip()
        if not prefix:
            raise ValueError("storage_prefix must be non-empty")
        if "/" in self.chunks_filename:
            raise ValueError("chunks_filename must not contain path separators")
        if "/" in self.embeddings_filename:
            raise ValueError("embeddings_filename must not contain path separators")
        if "/" in self.manifest_filename:
            raise ValueError("manifest_filename must not contain path separators")


@dataclass(frozen=True)
class ChunkPersistenceItem:
    sequence_index: int
    token_count: int
    chunk_checksum: str
    embedding_checksum: str


@dataclass(frozen=True)
class ChunkPersistenceResult:
    chunk_key: str
    embedding_key: str
    manifest_key: str
    chunk_checksum: str
    embedding_checksum: str
    items: Sequence[ChunkPersistenceItem]


class ChunkPersistence:
    """Persists chunk and embedding payloads alongside an integrity manifest."""

    def __init__(self, object_store: ObjectStoreClient, *, config: Optional[ChunkPersistenceConfig] = None) -> None:
        self._object_store = object_store
        self._config = config or ChunkPersistenceConfig()

    def persist(
        self,
        *,
        document_id: str,
        profile_hash: str,
        chunks: Sequence[ChunkCandidate],
        embeddings: EmbeddingExecutionResult,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ChunkPersistenceResult:
        if not chunks:
            raise ServiceError(
                error_code="chunk_persistence_empty",
                message="Cannot persist empty chunk set",
                http_status=422,
            )
        if len(chunks) != len(embeddings.embeddings):
            raise ServiceError(
                error_code="chunk_embedding_count_mismatch",
                message="Chunk count does not match embedding count",
                http_status=422,
            )
        expected_indices = [chunk.sequence_index for chunk in chunks]
        if embeddings.sequence_indices and list(embeddings.sequence_indices) != expected_indices:
            raise ServiceError(
                error_code="chunk_embedding_index_mismatch",
                message="Embedding sequence indices do not align with chunk order",
                http_status=422,
            )

        storage_prefix = self._build_prefix(document_id=document_id, profile_hash=profile_hash)
        chunk_key = f"{storage_prefix}/{self._config.chunks_filename}"
        embedding_key = f"{storage_prefix}/{self._config.embeddings_filename}"
        manifest_key = f"{storage_prefix}/{self._config.manifest_filename}"

        chunk_records, chunk_checksums = self._build_chunk_records(chunks)
        chunk_payload = self._encode_json_lines(chunk_records)
        chunk_checksum = self._object_store.upload_bytes(
            bucket=self._config.bucket,
            key=chunk_key,
            payload=chunk_payload,
            expected_checksum=compute_checksum(chunk_payload),
        )

        embedding_records, embedding_checksums = self._build_embedding_records(chunks, embeddings.embeddings)
        embedding_payload = self._encode_json_lines(embedding_records)
        embedding_checksum = self._object_store.upload_bytes(
            bucket=self._config.bucket,
            key=embedding_key,
            payload=embedding_payload,
            expected_checksum=compute_checksum(embedding_payload),
        )

        manifest_payload, manifest_items = self._build_manifest_payload(
            document_id=document_id,
            profile_hash=profile_hash,
            chunk_key=chunk_key,
            embedding_key=embedding_key,
            chunks=chunks,
            chunk_checksum=chunk_checksum,
            embedding_checksum=embedding_checksum,
            chunk_hashes=chunk_checksums,
            embedding_hashes=embedding_checksums,
            metadata=metadata,
        )
        self._object_store.upload_bytes(
            bucket=self._config.bucket,
            key=manifest_key,
            payload=manifest_payload,
            expected_checksum=compute_checksum(manifest_payload),
        )

        logger.info(
            "Persisted chunks and embeddings",
            extra={
                "code": "CHUNK_PERSIST_COMPLETED",
                "document_id": document_id,
                "profile_hash": profile_hash,
                "chunk_count": len(chunks),
                "chunk_key": chunk_key,
                "embedding_key": embedding_key,
                "manifest_key": manifest_key,
            },
        )
        return ChunkPersistenceResult(
            chunk_key=chunk_key,
            embedding_key=embedding_key,
            manifest_key=manifest_key,
            chunk_checksum=chunk_checksum,
            embedding_checksum=embedding_checksum,
            items=manifest_items,
        )

    def _build_prefix(self, *, document_id: str, profile_hash: str) -> str:
        safe_document_id = document_id.replace("/", "_")
        safe_profile_hash = profile_hash.replace("/", "_")
        prefix = self._config.storage_prefix.strip("/")
        return f"{prefix}/{safe_document_id}/{safe_profile_hash}"

    @staticmethod
    def _encode_json_lines(records: Iterable[Dict[str, object]]) -> bytes:
        serialized_lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
        return "\n".join(serialized_lines).encode("utf-8")

    @staticmethod
    def _build_chunk_records(chunks: Sequence[ChunkCandidate]) -> tuple[List[Dict[str, object]], List[str]]:
        records: List[Dict[str, object]] = []
        hashes: List[str] = []
        for chunk in chunks:
            checksum = compute_checksum(chunk.text.encode("utf-8"))
            hashes.append(checksum)
            records.append(
                {
                    "sequence_index": chunk.sequence_index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "text_checksum": checksum,
                }
            )
        return records, hashes

    @staticmethod
    def _build_embedding_records(
        chunks: Sequence[ChunkCandidate], embeddings: Sequence[Sequence[float]]
    ) -> tuple[List[Dict[str, object]], List[str]]:
        records: List[Dict[str, object]] = []
        hashes: List[str] = []
        for chunk, vector in zip(chunks, embeddings):
            record = {
                "sequence_index": chunk.sequence_index,
                "embedding": list(vector),
            }
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            checksum = compute_checksum(payload.encode("utf-8"))
            hashes.append(checksum)
            records.append(record)
        return records, hashes

    def _build_manifest_payload(
        self,
        *,
        document_id: str,
        profile_hash: str,
        chunk_key: str,
        embedding_key: str,
        chunks: Sequence[ChunkCandidate],
        chunk_checksum: str,
        embedding_checksum: str,
        chunk_hashes: Sequence[str],
        embedding_hashes: Sequence[str],
        metadata: Optional[Dict[str, str]],
    ) -> tuple[bytes, List[ChunkPersistenceItem]]:
        created_at = datetime.now(timezone.utc).isoformat()
        items: List[ChunkPersistenceItem] = []
        for chunk, chunk_hash, embedding_hash in zip(chunks, chunk_hashes, embedding_hashes):
            items.append(
                ChunkPersistenceItem(
                    sequence_index=chunk.sequence_index,
                    token_count=chunk.token_count,
                    chunk_checksum=chunk_hash,
                    embedding_checksum=embedding_hash,
                )
            )
        manifest = {
            "document_id": document_id,
            "profile_hash": profile_hash,
            "created_at": created_at,
            "chunk_key": chunk_key,
            "embedding_key": embedding_key,
            "chunk_checksum": chunk_checksum,
            "embedding_checksum": embedding_checksum,
            "counts": {
                "chunks": len(chunks),
                "embeddings": len(embedding_hashes),
            },
            "items": [
                {
                    "sequence_index": item.sequence_index,
                    "token_count": item.token_count,
                    "chunk_checksum": item.chunk_checksum,
                    "embedding_checksum": item.embedding_checksum,
                }
                for item in items
            ],
        }
        if metadata:
            manifest["metadata"] = dict(metadata)
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return manifest_bytes, items


__all__ = [
    "ChunkPersistence",
    "ChunkPersistenceConfig",
    "ChunkPersistenceItem",
    "ChunkPersistenceResult",
]
