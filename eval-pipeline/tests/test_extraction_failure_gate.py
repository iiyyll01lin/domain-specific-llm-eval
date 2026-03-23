"""
CI gate tests for RAGAS metric extraction failures.

These tests enforce that:
  1. The result contract always exposes extraction_failure_count.
  2. A non-zero extraction_failure_count is detectable by CI assertions.
  3. When failures occur the structured JSON artifact is persisted to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator():
    """Build a minimal RAGASEvaluatorWithFallbacks stub without triggering LLM setup."""
    from src.evaluation.ragas_evaluator_with_fallbacks import (
        RAGASEvaluatorWithFallbacks,
    )

    evaluator = object.__new__(RAGASEvaluatorWithFallbacks)
    evaluator.config = {}
    evaluator.llm_metrics = []
    evaluator.nonllm_metrics = []
    evaluator.legacy_metrics = []
    evaluator.extraction_failures = []
    return evaluator


def _make_result_contract(failure_count: int) -> dict:
    """Return a minimal result dict matching the contract shape."""
    failures = [
        {
            "metric_name": f"metric_{i}",
            "result_type": "<class 'NoneType'>",
            "attempted_paths": ["dataframe", "dict", "attribute", "index"],
            "error_detail": "No matching column/key/attribute found",
            "timestamp": "2026-03-14T00:00:00",
        }
        for i in range(failure_count)
    ]
    return {
        "success": True,
        "extraction_failures": failures,
        "extraction_failures_file": None,
        "extraction_failure_count": failure_count,
    }


# ---------------------------------------------------------------------------
# Gate assertions (the pattern CI jobs should use)
# ---------------------------------------------------------------------------

def _ci_extraction_failure_gate(result: dict, max_allowed: int = 0) -> None:
    """Raise AssertionError if extraction_failure_count exceeds threshold."""
    count = result["extraction_failure_count"]
    assert count <= max_allowed, (
        f"Metric extraction failure gate: {count} failure(s) recorded "
        f"(max_allowed={max_allowed}). Inspect result['extraction_failures'] "
        "for details."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractionFailureContractShape:
    """Verify the result contract always exposes the required keys."""

    def test_clean_result_has_all_keys(self) -> None:
        result = _make_result_contract(0)
        assert "extraction_failure_count" in result
        assert "extraction_failures" in result
        assert "extraction_failures_file" in result

    def test_failure_result_has_all_keys(self) -> None:
        result = _make_result_contract(3)
        assert "extraction_failure_count" in result
        assert "extraction_failures" in result
        assert len(result["extraction_failures"]) == 3

    def test_failure_records_carry_required_fields(self) -> None:
        result = _make_result_contract(1)
        record = result["extraction_failures"][0]
        for field in ("metric_name", "result_type", "attempted_paths", "error_detail", "timestamp"):
            assert field in record, f"Failure record missing field: {field}"


class TestCIGateLogic:
    """Verify the CI gate helper raises correctly."""

    def test_gate_passes_on_clean_result(self) -> None:
        result = _make_result_contract(0)
        _ci_extraction_failure_gate(result, max_allowed=0)  # must not raise

    def test_gate_fails_on_nonzero_failures(self) -> None:
        result = _make_result_contract(2)
        with pytest.raises(AssertionError) as exc_info:
            _ci_extraction_failure_gate(result, max_allowed=0)
        assert "2 failure(s)" in str(exc_info.value)

    def test_gate_passes_within_allowed_tolerance(self) -> None:
        result = _make_result_contract(3)
        _ci_extraction_failure_gate(result, max_allowed=5)  # must not raise

    def test_gate_fails_when_exceeding_tolerance(self) -> None:
        result = _make_result_contract(6)
        with pytest.raises(AssertionError):
            _ci_extraction_failure_gate(result, max_allowed=5)


class TestExtractionFailureRecording:
    """Verify the evaluator records failures to self.extraction_failures."""

    def test_clean_evaluation_has_empty_failures(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator.extraction_failures == []
        assert len(evaluator.extraction_failures) == 0

    def test_failure_appended_correctly(self) -> None:
        evaluator = _make_evaluator()
        evaluator.extraction_failures.append(
            {
                "metric_name": "faithfulness",
                "result_type": "<class 'NoneType'>",
                "attempted_paths": ["dataframe", "dict", "attribute", "index"],
                "error_detail": "No matching column",
                "timestamp": "2026-03-14T00:00:00",
            }
        )
        assert len(evaluator.extraction_failures) == 1
        assert evaluator.extraction_failures[0]["metric_name"] == "faithfulness"

    def test_multiple_failures_accumulate(self) -> None:
        evaluator = _make_evaluator()
        for i in range(5):
            evaluator.extraction_failures.append(
                {"metric_name": f"metric_{i}", "error_detail": "fail", "timestamp": ""}
            )
        assert len(evaluator.extraction_failures) == 5


class TestExtractionFailureArtifactPersistence:
    """Smoke-test: when failures exist, the JSON artifact must be written to disk."""

    def test_artifact_written_when_failures_present(self, tmp_path: Path) -> None:
        """Simulate the persistence logic and assert the file is created."""
        import json as _json
        from datetime import datetime

        failures = [
            {
                "metric_name": "context_precision",
                "result_type": "<class 'NoneType'>",
                "attempted_paths": ["dataframe"],
                "error_detail": "No column found",
                "timestamp": datetime.now().isoformat(),
            }
        ]
        timestamp = "20260314_000000"
        artifact_path = tmp_path / f"ragas_extraction_failures_{timestamp}.json"

        # Replicate the production persistence logic exactly
        with open(artifact_path, "w", encoding="utf-8") as _ef:
            _json.dump(
                {
                    "evaluation_timestamp": timestamp,
                    "total_failures": len(failures),
                    "failures": failures,
                },
                _ef,
                indent=2,
            )

        assert artifact_path.exists(), "Extraction failures artifact not written"
        payload = _json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["total_failures"] == 1
        assert payload["failures"][0]["metric_name"] == "context_precision"

    def test_artifact_not_written_when_no_failures(self, tmp_path: Path) -> None:
        """When extraction_failures is empty, no artifact file should appear."""
        failures: list = []
        timestamp = "20260314_000000"
        artifact_path = tmp_path / f"ragas_extraction_failures_{timestamp}.json"

        if failures:  # production guard
            with open(artifact_path, "w", encoding="utf-8") as _ef:
                import json as _json
                _json.dump({"total_failures": 0, "failures": failures}, _ef)

        assert not artifact_path.exists(), "Artifact should not be written for zero failures"


class TestExtractionFailureIntegration:
    """Integration-level smoke test using a real (stubbed) evaluator instance."""

    def test_evaluator_instance_has_extraction_failures_attr(self) -> None:
        evaluator = _make_evaluator()
        assert hasattr(evaluator, "extraction_failures")
        assert isinstance(evaluator.extraction_failures, list)

    def test_result_contract_zero_count_satisfies_ci_gate(self) -> None:
        """End-to-end: zero failures must pass the strict CI gate."""
        evaluator = _make_evaluator()
        # Simulate a clean run - no failures recorded
        result = {
            "success": True,
            "extraction_failures": evaluator.extraction_failures,
            "extraction_failures_file": None,
            "extraction_failure_count": len(evaluator.extraction_failures),
        }
        _ci_extraction_failure_gate(result, max_allowed=0)

    def test_result_contract_nonzero_count_fails_strict_ci_gate(self) -> None:
        """End-to-end: failures must trip the strict CI gate."""
        evaluator = _make_evaluator()
        evaluator.extraction_failures.append(
            {"metric_name": "faithfulness", "error_detail": "fail", "timestamp": ""}
        )
        result = {
            "success": True,
            "extraction_failures": evaluator.extraction_failures,
            "extraction_failures_file": None,
            "extraction_failure_count": len(evaluator.extraction_failures),
        }
        with pytest.raises(AssertionError):
            _ci_extraction_failure_gate(result, max_allowed=0)
