from __future__ import annotations

import pytest

from src.evaluation.ragas_bypass import RAGASBypass


def test_ragas_bypass_marks_mock_results_explicitly() -> None:
    result = RAGASBypass.generate_mock_ragas_result(metrics=[])
    payload = result.to_dict()

    assert payload["is_mock_result"] is True
    assert payload["result_source"] == "ragas_bypass"
    assert payload["fallback_reason"] == "ragas_bypass_enabled"


def test_ragas_bypass_missing_metric_raises_key_error() -> None:
    result = RAGASBypass.generate_mock_ragas_result(metrics=[])

    with pytest.raises(KeyError):
        _ = result["missing_metric"]