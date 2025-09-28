from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Protocol, Sequence

from prometheus_client import Counter, Histogram

from services.common.config import settings
from services.common.errors import ServiceError
from services.processing.stages.chunk_rules import ChunkCandidate

logger = logging.getLogger(__name__)

_EMBEDDING_DURATION = Histogram(
    "processing_embedding_batch_duration_seconds",
    "Processing embedding batch duration in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

_EMBEDDING_BATCH_SIZE = Histogram(
    "processing_embedding_batch_size",
    "Embedding batch size distribution",
    buckets=(1, 8, 16, 32, 64, 128, 256, 512, 1024),
)

_EMBEDDING_ERRORS = Counter(
    "processing_embedding_error_total",
    "Total number of embedding batch errors",
)


class EmbeddingMetricsRecorder(Protocol):
    def observe_duration(self, seconds: float) -> None:
        ...

    def observe_batch_size(self, batch_size: int) -> None:
        ...

    def increment_errors(self) -> None:
        ...


class PrometheusEmbeddingMetrics(EmbeddingMetricsRecorder):
    def __init__(
        self,
        duration_histogram: Histogram = _EMBEDDING_DURATION,
        batch_histogram: Histogram = _EMBEDDING_BATCH_SIZE,
        error_counter: Counter = _EMBEDDING_ERRORS,
    ) -> None:
        self._duration = duration_histogram
        self._batch = batch_histogram
        self._errors = error_counter

    def observe_duration(self, seconds: float) -> None:
        self._duration.observe(max(0.0, seconds))

    def observe_batch_size(self, batch_size: int) -> None:
        self._batch.observe(max(0, batch_size))

    def increment_errors(self) -> None:
        self._errors.inc()


@dataclass(frozen=True)
class EmbeddingExecutorConfig:
    batch_size: int = 512
    max_retries: int = 3
    base_backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0
    breaker_threshold: int = 5
    breaker_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self.base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.breaker_threshold <= 0:
            raise ValueError("breaker_threshold must be positive")
        if self.breaker_cooldown_seconds <= 0:
            raise ValueError("breaker_cooldown_seconds must be positive")


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str], *, timeout: float) -> Sequence[Sequence[float]]:
        ...


class EmbeddingProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class EmbeddingExecutionResult:
    embeddings: List[List[float]]
    sequence_indices: List[int]


class EmbeddingBatchExecutor:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        config: EmbeddingExecutorConfig | None = None,
        metrics: EmbeddingMetricsRecorder | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._config = config or EmbeddingExecutorConfig()
        self._metrics = metrics or PrometheusEmbeddingMetrics()
        self._sleep = sleep_func
        self._clock = monotonic
        self._consecutive_failures = 0
        self._breaker_open_until: float | None = None

    def execute(self, chunks: Sequence[ChunkCandidate]) -> EmbeddingExecutionResult:
        if not chunks:
            return EmbeddingExecutionResult(embeddings=[], sequence_indices=[])
        embeddings: List[List[float]] = []
        indices: List[int] = []
        max_batch_size = max(
            1,
            min(self._config.batch_size, settings.processing_embedding_max_batch_size),
        )
        if len(chunks) > max_batch_size:
            logger.info(
                "Embedding batch split due to max batch size",
                extra={
                    "code": "EMBED_BATCH_SPLIT",
                    "max_batch_size": max_batch_size,
                    "total": len(chunks),
                },
            )
        for offset in range(0, len(chunks), max_batch_size):
            batch_chunks = list(chunks[offset : offset + max_batch_size])
            self._ensure_breaker_closed()
            texts = [chunk.text for chunk in batch_chunks]
            vectors = self._execute_batch(texts, len(batch_chunks))
            if len(vectors) != len(batch_chunks):
                raise ServiceError(
                    error_code="embedding_mismatch",
                    message="Embedding provider returned unexpected vector count",
                    http_status=422,
                )
            embeddings.extend([list(vector) for vector in vectors])
            indices.extend([chunk.sequence_index for chunk in batch_chunks])
        return EmbeddingExecutionResult(embeddings=embeddings, sequence_indices=indices)

    def _execute_batch(self, texts: Sequence[str], batch_size: int) -> Sequence[Sequence[float]]:
        start_time = self._clock()
        logger.info(
            "Embedding batch start",
            extra={"code": "EMBED_BATCH_START", "batch_size": batch_size},
        )
        vectors = self._call_provider_with_retry(texts, batch_size)
        duration = self._clock() - start_time
        self._metrics.observe_duration(duration)
        self._metrics.observe_batch_size(batch_size)
        self._consecutive_failures = 0
        return vectors

    def _call_provider_with_retry(
        self, texts: Sequence[str], batch_size: int
    ) -> Sequence[Sequence[float]]:
        attempts = self._config.max_retries
        for attempt in range(1, attempts + 1):
            try:
                return self._provider.embed(texts, timeout=self._config.timeout_seconds)
            except EmbeddingProviderError as exc:
                self._metrics.increment_errors()
                breaker_opened = self._register_failure(
                    exc, retryable=exc.retryable, attempt=attempt, batch_size=batch_size
                )
                if not exc.retryable or attempt == attempts or breaker_opened:
                    raise self._wrap_error(exc)
                self._apply_backoff(attempt)
            except Exception as exc:
                self._metrics.increment_errors()
                self._register_failure(exc, retryable=False, attempt=attempt, batch_size=batch_size)
                raise self._wrap_error(exc)
        raise self._wrap_error(RuntimeError("embedding attempts exhausted"))

    def _apply_backoff(self, attempt: int) -> None:
        delay = self._config.base_backoff_seconds * math.pow(2, attempt - 1)
        if delay > 0:
            self._sleep(delay)

    def _register_failure(
        self,
        exc: Exception,
        *,
        retryable: bool,
        attempt: int,
        batch_size: int,
    ) -> bool:
        self._consecutive_failures += 1
        logger.error(
            "Embedding batch failure",
            extra={
                "code": "EMBED_BATCH_FAIL",
                "retryable": retryable,
                "attempt": attempt,
                "batch_size": batch_size,
            },
        )
        if self._consecutive_failures >= self._config.breaker_threshold:
            self._breaker_open_until = self._clock() + self._config.breaker_cooldown_seconds
            logger.warning(
                "Embedding circuit breaker opened",
                extra={
                    "code": "EMBED_BREAKER_OPEN",
                    "cooldown_seconds": self._config.breaker_cooldown_seconds,
                    "failure_count": self._consecutive_failures,
                },
            )
            return True
        return False

    def _ensure_breaker_closed(self) -> None:
        if self._breaker_open_until is None:
            return
        now = self._clock()
        if now < self._breaker_open_until:
            raise ServiceError(
                error_code="embedding_circuit_open",
                message="Embedding provider temporarily unavailable",
                http_status=503,
            )
        self._breaker_open_until = None
        self._consecutive_failures = 0

    @staticmethod
    def _wrap_error(exc: Exception) -> ServiceError:
        return ServiceError(
            error_code="embedding_provider_unavailable",
            message=str(exc),
            http_status=502,
        )


__all__ = [
    "EmbeddingBatchExecutor",
    "EmbeddingExecutionResult",
    "EmbeddingExecutorConfig",
    "EmbeddingMetricsRecorder",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "PrometheusEmbeddingMetrics",
]
