from __future__ import annotations

import json
import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.errors import ServiceError  # noqa: E402
from services.common.storage.object_store import compute_checksum  # noqa: E402
from services.processing.stages import (  # noqa: E402
    ChunkCandidate,
    ChunkPersistence,
    ChunkPersistenceConfig,
    EmbeddingExecutionResult,
)


class FakeObjectStore:
    def __init__(self) -> None:
        self.uploads: List[dict] = []

    def upload_bytes(self, bucket, key, payload, expected_checksum=None):  # type: ignore[no-untyped-def]
        checksum = compute_checksum(payload)
        if expected_checksum and expected_checksum != checksum:
            raise AssertionError("Expected checksum mismatch in test harness")
        self.uploads.append(
            {
                "bucket": bucket,
                "key": key,
                "payload": payload,
                "checksum": checksum,
            }
        )
        return checksum


def _make_chunks(count: int) -> List[ChunkCandidate]:
    return [
        ChunkCandidate(
            sequence_index=index,
            text=f"chunk-{index}",
            token_count=10 + index,
            tokens=(f"tok-{index}",),
        )
        for index in range(count)
    ]


def test_chunk_persistence_uploads_files_and_manifest() -> None:
    object_store = FakeObjectStore()
    persistence = ChunkPersistence(
        object_store,
        config=ChunkPersistenceConfig(bucket="unit-test", storage_prefix="chunks-store"),
    )
    chunks = _make_chunks(2)
    embeddings = EmbeddingExecutionResult(embeddings=[[0.1, 0.2], [0.3, 0.4]], sequence_indices=[0, 1])

    result = persistence.persist(
        document_id="doc-123",
        profile_hash="profile-abc",
        chunks=chunks,
        embeddings=embeddings,
        metadata={"km_id": "KM-001"},
    )

    assert len(object_store.uploads) == 3
    chunk_upload, embedding_upload, manifest_upload = object_store.uploads

    assert chunk_upload["bucket"] == "unit-test"
    assert chunk_upload["key"] == "chunks-store/doc-123/profile-abc/chunks.jsonl"
    chunk_lines = chunk_upload["payload"].decode("utf-8").splitlines()
    assert len(chunk_lines) == 2

    assert embedding_upload["key"] == "chunks-store/doc-123/profile-abc/embeddings.jsonl"
    embedding_lines = embedding_upload["payload"].decode("utf-8").splitlines()
    assert len(embedding_lines) == 2

    manifest = json.loads(manifest_upload["payload"].decode("utf-8"))
    assert manifest["counts"] == {"chunks": 2, "embeddings": 2}
    assert manifest["chunk_key"].endswith("chunks.jsonl")
    assert manifest["embedding_key"].endswith("embeddings.jsonl")
    assert manifest["chunk_checksum"] == chunk_upload["checksum"]
    assert manifest["embedding_checksum"] == embedding_upload["checksum"]
    assert manifest["metadata"] == {"km_id": "KM-001"}
    assert len(manifest["items"]) == 2
    for item, chunk in zip(manifest["items"], chunks):
        expected_chunk_checksum = compute_checksum(chunk.text.encode("utf-8"))
        assert item["sequence_index"] == chunk.sequence_index
        assert item["token_count"] == chunk.token_count
        assert item["chunk_checksum"] == expected_chunk_checksum

    assert result.chunk_key == chunk_upload["key"]
    assert result.embedding_key == embedding_upload["key"]
    assert result.manifest_key == manifest_upload["key"]
    assert result.chunk_checksum == chunk_upload["checksum"]
    assert result.embedding_checksum == embedding_upload["checksum"]
    assert len(result.items) == 2


def test_chunk_persistence_raises_on_count_mismatch() -> None:
    object_store = FakeObjectStore()
    persistence = ChunkPersistence(object_store)
    chunks = _make_chunks(2)
    embeddings = EmbeddingExecutionResult(embeddings=[[0.1, 0.2]], sequence_indices=[0])

    with pytest.raises(ServiceError) as exc:
        persistence.persist(
            document_id="doc-1",
            profile_hash="profile",
            chunks=chunks,
            embeddings=embeddings,
        )

    assert exc.value.error_code == "chunk_embedding_count_mismatch"


def test_chunk_persistence_raises_on_sequence_mismatch() -> None:
    object_store = FakeObjectStore()
    persistence = ChunkPersistence(object_store)
    chunks = _make_chunks(2)
    embeddings = EmbeddingExecutionResult(embeddings=[[0.1], [0.2]], sequence_indices=[0, 3])

    with pytest.raises(ServiceError) as exc:
        persistence.persist(
            document_id="doc-1",
            profile_hash="profile",
            chunks=chunks,
            embeddings=embeddings,
        )

    assert exc.value.error_code == "chunk_embedding_index_mismatch"
