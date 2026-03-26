from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Sequence

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.config import settings  # noqa: E402
from services.common.errors import ServiceError  # noqa: E402
from services.processing.stages import (  # noqa: E402
    ChunkCandidate,
    EmbeddingBatchExecutor,
    EmbeddingExecutorConfig,
    EmbeddingProviderError,
)


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class DummyMetrics:
    def __init__(self) -> None:
        self.durations: List[float] = []
        self.batch_sizes: List[int] = []
        self.errors: int = 0

    def observe_duration(self, seconds: float) -> None:
        self.durations.append(seconds)

    def observe_batch_size(self, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)

    def increment_errors(self) -> None:
        self.errors += 1


class StubProvider:
    def __init__(
        self,
        *,
        clock: FakeClock,
        vectors: Sequence[Sequence[float]] | None = None,
        retryable_failures: int = 0,
        non_retryable_failures: int = 0,
    ) -> None:
        self.clock = clock
        self.vectors = [list(vec) for vec in vectors or []]
        self.retryable_failures = retryable_failures
        self.non_retryable_failures = non_retryable_failures
        self.calls: int = 0

    def embed(self, texts: Sequence[str], *, timeout: float):
        self.calls += 1
        if self.non_retryable_failures > 0:
            self.non_retryable_failures -= 1
            raise EmbeddingProviderError("hard failure", retryable=False)
        if self.retryable_failures > 0:
            self.retryable_failures -= 1
            raise EmbeddingProviderError("transient failure", retryable=True)
        self.clock.advance(0.2)
        if self.vectors:
            return [list(vec) for vec in self.vectors[: len(texts)]]
        return [[float(index)] for index, _ in enumerate(texts)]


def _make_chunks(count: int) -> List[ChunkCandidate]:
    return [
        ChunkCandidate(
            sequence_index=index,
            text=f"chunk-{index}",
            token_count=index + 1,
            tokens=(f"tok-{index}",),
        )
        for index in range(count)
    ]


def test_embedding_executor_success() -> None:
    clock = FakeClock()
    metrics = DummyMetrics()
    provider = StubProvider(clock=clock, vectors=[[1.0], [2.0]])
    executor = EmbeddingBatchExecutor(
        provider,
        metrics=metrics,
        monotonic=clock,
        sleep_func=clock.advance,
    )

    result = executor.execute(_make_chunks(2))

    assert result.embeddings == [[1.0], [2.0]]
    assert result.sequence_indices == [0, 1]
    assert provider.calls == 1
    assert metrics.batch_sizes == [2]
    assert metrics.errors == 0
    assert metrics.durations[0] == pytest.approx(0.2, abs=1e-6)


def test_embedding_executor_retries_transient_failures() -> None:
    clock = FakeClock()
    metrics = DummyMetrics()
    provider = StubProvider(clock=clock, retryable_failures=2, vectors=[[0.1], [0.2]])
    executor = EmbeddingBatchExecutor(
        provider,
        metrics=metrics,
        monotonic=clock,
        sleep_func=clock.advance,
        config=EmbeddingExecutorConfig(max_retries=3, batch_size=4),
    )

    result = executor.execute(_make_chunks(2))

    assert result.embeddings == [[0.1], [0.2]]
    assert provider.calls == 3
    assert metrics.errors == 2
    assert len(metrics.durations) == 1


def test_embedding_executor_opens_circuit_after_consecutive_failures() -> None:
    clock = FakeClock()
    metrics = DummyMetrics()
    provider = StubProvider(clock=clock, retryable_failures=100)
    executor = EmbeddingBatchExecutor(
        provider,
        metrics=metrics,
        monotonic=clock,
        sleep_func=clock.advance,
        config=EmbeddingExecutorConfig(max_retries=1, breaker_threshold=2, breaker_cooldown_seconds=10),
    )

    with pytest.raises(ServiceError) as first_error:
        executor.execute(_make_chunks(1))
    assert first_error.value.error_code == "embedding_provider_unavailable"
    assert provider.calls == 1

    with pytest.raises(ServiceError) as second_error:
        executor.execute(_make_chunks(1))
    assert second_error.value.error_code == "embedding_provider_unavailable"
    assert provider.calls == 2

    with pytest.raises(ServiceError) as breaker_error:
        executor.execute(_make_chunks(1))
    assert breaker_error.value.error_code == "embedding_circuit_open"
    assert provider.calls == 2


def test_embedding_executor_handles_non_retryable_error() -> None:
    clock = FakeClock()
    metrics = DummyMetrics()
    provider = StubProvider(clock=clock, non_retryable_failures=1)
    executor = EmbeddingBatchExecutor(
        provider,
        metrics=metrics,
        monotonic=clock,
        sleep_func=clock.advance,
        config=EmbeddingExecutorConfig(max_retries=5),
    )

    with pytest.raises(ServiceError) as exc:
        executor.execute(_make_chunks(1))
    assert exc.value.error_code == "embedding_provider_unavailable"
    assert provider.calls == 1
    assert metrics.errors == 1


def test_embedding_executor_respects_settings_max_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = settings.processing_embedding_max_batch_size
    monkeypatch.setattr(settings, "processing_embedding_max_batch_size", 1)
    clock = FakeClock()
    metrics = DummyMetrics()
    provider = StubProvider(clock=clock)
    executor = EmbeddingBatchExecutor(
        provider,
        metrics=metrics,
        monotonic=clock,
        sleep_func=clock.advance,
        config=EmbeddingExecutorConfig(batch_size=10),
    )

    result = executor.execute(_make_chunks(3))

    assert len(result.embeddings) == 3
    assert provider.calls == 3
    assert metrics.batch_sizes == [1, 1, 1]