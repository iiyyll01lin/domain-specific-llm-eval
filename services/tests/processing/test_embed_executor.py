from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pytest

from services.common.config import settings
from services.common.errors import ServiceError
from services.processing.stages.chunk_rules import ChunkCandidate
from services.processing.stages.embed_executor import (
    EmbeddingBatchExecutor,
    EmbeddingExecutionResult,
    EmbeddingExecutorConfig,
    EmbeddingMetricsRecorder,
    EmbeddingProvider,
    EmbeddingProviderError,
)


@dataclass
class RecordingMetrics(EmbeddingMetricsRecorder):
    durations: List[float]
    batch_sizes: List[int]
    error_count: int = 0

    def observe_duration(self, seconds: float) -> None:
        self.durations.append(seconds)

    def observe_batch_size(self, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)

    def increment_errors(self) -> None:
        self.error_count += 1


class FailingProvider(EmbeddingProvider):
    def __init__(self, *, retryable: bool, max_failures: int) -> None:
        self.retryable = retryable
        self.max_failures = max_failures
        self.calls = 0

    def embed(self, texts: Sequence[str], *, timeout: float) -> Sequence[Sequence[float]]:
        self.calls += 1
        if self.calls <= self.max_failures:
            raise EmbeddingProviderError("rate limited", retryable=self.retryable)
        return [[float(index)] for index, _ in enumerate(texts)]


class CountingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.batch_sizes: List[int] = []

    def embed(self, texts: Sequence[str], *, timeout: float) -> Sequence[Sequence[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(index)] for index, _ in enumerate(texts)]


class MismatchedProvider(EmbeddingProvider):
    def embed(self, texts: Sequence[str], *, timeout: float) -> Sequence[Sequence[float]]:
        return []


@pytest.fixture
def sample_chunks() -> Sequence[ChunkCandidate]:
    return [
        ChunkCandidate(sequence_index=0, text="alpha", token_count=1, tokens=("alpha",)),
        ChunkCandidate(sequence_index=1, text="beta", token_count=1, tokens=("beta",)),
        ChunkCandidate(sequence_index=2, text="gamma", token_count=1, tokens=("gamma",)),
    ]


def test_executor_retries_and_raises_after_final_failure(sample_chunks: Sequence[ChunkCandidate]) -> None:
    metrics = RecordingMetrics(durations=[], batch_sizes=[])
    provider = FailingProvider(retryable=True, max_failures=3)
    executor = EmbeddingBatchExecutor(
        provider,
        config=EmbeddingExecutorConfig(max_retries=3, base_backoff_seconds=0.0, breaker_threshold=4),
        metrics=metrics,
        sleep_func=lambda _: None,
    )

    with pytest.raises(ServiceError) as exc_info:
        executor.execute(sample_chunks)

    assert exc_info.value.error_code == "embedding_provider_unavailable"
    assert provider.calls == 3
    assert metrics.error_count == 3


def test_executor_respects_processing_batch_limit(
    sample_chunks: Sequence[ChunkCandidate], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "processing_embedding_max_batch_size", 2)
    metrics = RecordingMetrics(durations=[], batch_sizes=[])
    provider = CountingProvider()
    executor = EmbeddingBatchExecutor(
        provider,
        config=EmbeddingExecutorConfig(batch_size=8),
        metrics=metrics,
    )

    result = executor.execute(sample_chunks)

    assert provider.batch_sizes == [2, 1]
    assert len(result.embeddings) == len(sample_chunks)
    assert result.sequence_indices == [0, 1, 2]


def test_executor_detects_embedding_count_mismatch(sample_chunks: Sequence[ChunkCandidate]) -> None:
    executor = EmbeddingBatchExecutor(
        MismatchedProvider(),
        config=EmbeddingExecutorConfig(max_retries=1),
        metrics=RecordingMetrics(durations=[], batch_sizes=[]),
    )

    with pytest.raises(ServiceError) as exc_info:
        executor.execute(sample_chunks)

    assert exc_info.value.error_code == "embedding_mismatch"
