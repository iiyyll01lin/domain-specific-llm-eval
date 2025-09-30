"""Retry strategy for RAG invocations (TASK-031c)."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RAGInvocationError(Exception):
    """Represents a transport or service failure when calling the RAG system."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code

    def is_retryable(self, retry_status_codes: Sequence[int]) -> bool:
        return self.status_code in retry_status_codes if self.status_code is not None else False


@dataclass(frozen=True)
class RetryTelemetry:
    attempts: int
    last_delay_seconds: float
    last_error: Optional[str]


class RetryPolicy:
    """Simple exponential backoff with jitter tailored for the evaluation service."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 5.0,
        jitter_fraction: float = 0.2,
        retry_status_codes: Optional[Iterable[int]] = None,
        retry_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._max_attempts = max_attempts
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._jitter_fraction = max(jitter_fraction, 0.0)
        self._retry_status_codes = tuple(retry_status_codes or (429, 500, 502, 503, 504))
        self._retry_exceptions = retry_exceptions or (TimeoutError,)
        self._sleep = sleep_fn

    @property
    def max_attempts(self) -> int:
        """Expose the configured maximum number of attempts."""

        return self._max_attempts

    def execute(self, operation: Callable[[], T]) -> Tuple[T, RetryTelemetry]:
        attempts = 0
        last_delay = 0.0
        last_error: Optional[str] = None

        while attempts < self._max_attempts:
            attempts += 1
            try:
                result = operation()
                return result, RetryTelemetry(attempts=attempts, last_delay_seconds=last_delay, last_error=last_error)
            except self._retry_exceptions as exc:  # type: ignore[misc]
                last_error = repr(exc)
                if attempts >= self._max_attempts:
                    logger.error(
                        "rag.retry_exhausted",
                        extra={
                            "context": {
                                "attempts": attempts,
                                "error": last_error,
                                "type": exc.__class__.__name__,
                            }
                        },
                    )
                    raise
                last_delay = self._compute_delay(attempts)
                self._sleep(last_delay)
                continue
            except RAGInvocationError as exc:
                last_error = repr(exc)
                if not exc.is_retryable(self._retry_status_codes) or attempts >= self._max_attempts:
                    logger.error(
                        "rag.retry_exhausted",
                        extra={
                            "context": {
                                "attempts": attempts,
                                "status_code": exc.status_code,
                                "error_code": exc.error_code,
                                "message": str(exc),
                            }
                        },
                    )
                    raise
                last_delay = self._compute_delay(attempts)
                logger.warning(
                    "rag.retry",
                    extra={
                        "context": {
                            "attempt": attempts,
                            "status_code": exc.status_code,
                            "delay_seconds": last_delay,
                        }
                    },
                )
                self._sleep(last_delay)
        # Loop should return or raise; this safeguard will never execute but keeps mypy happy.
        raise RuntimeError("RetryPolicy exhausted without returning or raising")

    def _compute_delay(self, attempt: int) -> float:
        exponent = attempt - 1
        base_delay = min(self._base_delay_seconds * (2 ** exponent), self._max_delay_seconds)
        jitter_span = base_delay * self._jitter_fraction
        if jitter_span == 0:
            return base_delay
        return base_delay + random.uniform(-jitter_span, jitter_span)


__all__ = ["RetryPolicy", "RAGInvocationError", "RetryTelemetry"]
