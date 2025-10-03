from __future__ import annotations

import math

from services.eval.aggregation.sanitizer import sanitize_metrics


def test_sanitize_metrics_handles_nan_and_inf() -> None:
    payload = {
        "faithfulness": 0.95,
        "answer_relevancy": float("nan"),
        "precision": float("inf"),
        "recall": None,
    }
    sanitized = sanitize_metrics(payload)
    assert sanitized["faithfulness"] == 0.95
    assert sanitized["answer_relevancy"] is None
    assert sanitized["precision"] is None
    assert sanitized["recall"] is None


def test_sanitize_metrics_preserves_numbers() -> None:
    payload = {"m1": 0.123, "m2": 0}
    sanitized = sanitize_metrics(payload)
    assert math.isclose(sanitized["m1"], 0.123)
    assert sanitized["m2"] == 0.0
