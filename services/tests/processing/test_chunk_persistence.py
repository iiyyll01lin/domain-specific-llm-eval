from __future__ import annotations

from typing import Dict, Tuple

import pytest

from services.common.errors import ServiceError
from services.processing.stages import ChunkPersistence, ChunkPersistenceConfig
from services.processing.stages.chunk_rules import ChunkCandidate
from services.processing.stages.embed_executor import EmbeddingExecutionResult


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.uploads: Dict[Tuple[str | None, str], bytes] = {}

    def upload_bytes(self, *, bucket, key, payload, expected_checksum=None):  # type: ignore[no-untyped-def]
        self.uploads[(bucket, key)] = payload
        return expected_checksum or "test-checksum"


def _build_chunks(count: int) -> list[ChunkCandidate]:
    return [
        ChunkCandidate(
            sequence_index=index,
            text=f"chunk-{index}",
            token_count=5,
            tokens=(f"token-{index}",),
        )
        for index in range(count)
    ]


def test_persist_rejects_mismatched_embedding_count() -> None:
    object_store = InMemoryObjectStore()
    persistence = ChunkPersistence(object_store)
    chunks = _build_chunks(3)
    embeddings = EmbeddingExecutionResult(embeddings=[[0.1], [0.2]], sequence_indices=[0, 1])

    with pytest.raises(ServiceError) as exc_info:
        persistence.persist(
            document_id="doc-123",
            profile_hash="profile-hash",
            chunks=chunks,
            embeddings=embeddings,
            metadata={"km_id": "KM-1"},
        )

    assert exc_info.value.error_code == "chunk_embedding_count_mismatch"


def test_persist_generates_manifest_with_matching_counts() -> None:
    object_store = InMemoryObjectStore()
    persistence = ChunkPersistence(object_store, config=ChunkPersistenceConfig(storage_prefix="test-prefix"))
    chunks = _build_chunks(2)
    embeddings = EmbeddingExecutionResult(embeddings=[[0.1], [0.2]], sequence_indices=[0, 1])

    result = persistence.persist(
        document_id="doc-789",
        profile_hash="hash",
        chunks=chunks,
        embeddings=embeddings,
        metadata={"km_id": "KM-1"},
    )

    assert result.manifest_key.endswith("/manifest.json")
    assert len(result.items) == 2
    assert {item.sequence_index for item in result.items} == {0, 1}
    assert {item.token_count for item in result.items} == {5}
    manifest_payload = object_store.uploads[(None, result.manifest_key)]
    assert b"\"counts\":{\"chunks\":2,\"embeddings\":2}" in manifest_payload
