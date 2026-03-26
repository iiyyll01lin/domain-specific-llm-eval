from __future__ import annotations

from typing import List

import pytest

from services.eval.retry_policy import RAGInvocationError, RetryPolicy


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: List[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_retry_policy_retries_rate_limit_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    recorder = SleepRecorder()
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.1,
        jitter_fraction=0.0,
        sleep_fn=recorder,
    )

    attempts = 0

    def failing_operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RAGInvocationError("rate limited", status_code=429)

    with pytest.raises(RAGInvocationError):
        policy.execute(failing_operation)

    assert attempts == 3
    assert recorder.delays == [0.1, 0.2]
    assert any(
        record.message == "rag.retry_exhausted" and record.levelname == "ERROR"
        for record in caplog.records
    )


def test_retry_policy_succeeds_after_timeout_retry() -> None:
    recorder = SleepRecorder()
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.05,
        jitter_fraction=0.0,
        sleep_fn=recorder,
    )

    attempts = 0

    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("socket timed out")
        return "ok"

    result, telemetry = policy.execute(flaky_operation)

    assert result == "ok"
    assert attempts == 2
    assert recorder.delays == [0.05]
    assert telemetry.attempts == 2
    assert telemetry.last_delay_seconds == 0.05
    assert "TimeoutError" in telemetry.last_error