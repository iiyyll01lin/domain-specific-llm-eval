"""
TDD tests — Option D: Offline deterministic metrics as the default CI evaluation tier.

These tests assert that:
1. RagasEvaluator instantiates NonLLM/BLEU/ROUGE metrics unconditionally (no LLM required).
2. When safe_ragas_evaluate returns None, the result is a real error (mock_data=False,
   success=False) — NOT a silent mock payload.
3. When an exception occurs in evaluate(), the result is also mock_data=False.
4. RAGASEvaluatorWithFallbacks always has offline metrics in its primary tier.
5. No LLM network calls are attempted when the pipeline is in offline/CI mode.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# RagasEvaluator — offline metric presence
# ---------------------------------------------------------------------------

def test_ragas_evaluator_uses_offline_metrics_when_no_llm_configured() -> None:
    """RagasEvaluator must include offline deterministic metrics even with empty config."""
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(config={})

    offline_class_names = {m.__class__.__name__ for m in evaluator.metrics}
    assert "NonLLMContextPrecisionWithReference" in offline_class_names, (
        f"Expected NonLLMContextPrecisionWithReference in metrics, got: {offline_class_names}"
    )
    assert "NonLLMContextRecall" in offline_class_names, (
        f"Expected NonLLMContextRecall in metrics, got: {offline_class_names}"
    )
    assert "BleuScore" in offline_class_names, (
        f"Expected BleuScore in metrics, got: {offline_class_names}"
    )
    assert "RougeScore" in offline_class_names, (
        f"Expected RougeScore in metrics, got: {offline_class_names}"
    )


def test_ragas_evaluator_no_llm_metrics_when_endpoint_absent() -> None:
    """Without an LLM endpoint, legacy LLM-based metrics must not appear in self.metrics."""
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(config={})

    llm_only_names = {"Faithfulness", "ContextPrecision", "ContextRecall", "AnswerRelevancy"}
    metric_class_names = {m.__class__.__name__ for m in evaluator.metrics}
    llm_metrics_present = metric_class_names & llm_only_names
    assert not llm_metrics_present, (
        f"LLM-based metrics appeared without endpoint config: {llm_metrics_present}"
    )


# ---------------------------------------------------------------------------
# RagasEvaluator.evaluate() — mock_data=True paths eliminated
# ---------------------------------------------------------------------------

def test_safe_evaluate_none_yields_error_not_mock(monkeypatch) -> None:
    """
    When RagasModelDumpFix.safe_ragas_evaluate returns None, the result contract
    must have mock_data=False and success=False (not a silent mock pass).
    """
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = object.__new__(RagasEvaluator)
    evaluator.ragas_available = True
    evaluator.metrics = []
    evaluator._prepare_evaluation_data = lambda testset, responses: {
        "question": ["Q1"],
        "answer": ["A1"],
        "contexts": [["C1"]],
        "ground_truth": ["G1"],
    }

    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.fix_ragas_dataset_format",
        lambda data: data,
    )
    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.create_safe_ragas_dataset",
        lambda data: type("_D", (), {
            "__len__": lambda self: 1,
            "column_names": ["question"],
        })(),
    )
    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.safe_ragas_evaluate",
        lambda dataset, metrics: None,
    )

    result = evaluator.evaluate({"questions": ["Q1"]}, [{"answer": "A1"}])

    assert result["mock_data"] is False, (
        "safe_ragas_evaluate returning None must NOT produce mock_data=True"
    )
    assert result["success"] is False, (
        "safe_ragas_evaluate returning None must produce success=False"
    )
    assert result["result_source"] == "ragas_safe_evaluate_failed"
    assert result["error_stage"] == "ragas_safe_evaluate"
    assert result["contract_version"]


def test_evaluate_exception_yields_error_not_mock(monkeypatch) -> None:
    """
    When safe_ragas_evaluate raises, the result contract must have
    mock_data=False and success=False (not a mock fallback).
    """
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = object.__new__(RagasEvaluator)
    evaluator.ragas_available = True
    evaluator.metrics = []
    evaluator._prepare_evaluation_data = lambda testset, responses: {
        "question": ["Q1"],
        "answer": ["A1"],
        "contexts": [["C1"]],
        "ground_truth": ["G1"],
    }

    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.fix_ragas_dataset_format",
        lambda data: data,
    )
    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.create_safe_ragas_dataset",
        lambda data: type("_D", (), {
            "__len__": lambda self: 1,
            "column_names": ["question"],
        })(),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("LLM endpoint unreachable — simulating CI offline run")

    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.safe_ragas_evaluate",
        _raise,
    )

    result = evaluator.evaluate({"questions": ["Q1"]}, [{"answer": "A1"}])

    assert result["mock_data"] is False, (
        "Exception in evaluate() must NOT produce mock_data=True"
    )
    assert result["success"] is False


# ---------------------------------------------------------------------------
# RAGASEvaluatorWithFallbacks — offline metrics as primary tier
# ---------------------------------------------------------------------------

def test_ragas_evaluator_with_fallbacks_has_offline_metrics_without_llm() -> None:
    """
    RAGASEvaluatorWithFallbacks must populate nonllm_metrics with deterministic
    offline metrics even when no LLM endpoint is configured.
    """
    from src.evaluation.ragas_evaluator_with_fallbacks import RAGASEvaluatorWithFallbacks

    evaluator = RAGASEvaluatorWithFallbacks(config={})

    offline_class_names = {m.__class__.__name__ for m in evaluator.nonllm_metrics}
    assert "NonLLMContextPrecisionWithReference" in offline_class_names, (
        f"Expected NonLLMContextPrecisionWithReference in nonllm_metrics, got: {offline_class_names}"
    )
    assert "NonLLMContextRecall" in offline_class_names, (
        f"Expected NonLLMContextRecall in nonllm_metrics, got: {offline_class_names}"
    )


def test_ragas_evaluator_with_fallbacks_no_llm_means_no_network_metrics() -> None:
    """
    Without an LLM endpoint, llm_metrics must be empty (no network calls attempted).
    """
    from src.evaluation.ragas_evaluator_with_fallbacks import RAGASEvaluatorWithFallbacks

    evaluator = RAGASEvaluatorWithFallbacks(config={})

    assert evaluator.llm is None, "No LLM should be configured without endpoint"
    assert evaluator.llm_metrics == [], (
        f"llm_metrics must be empty without LLM, got: {evaluator.llm_metrics}"
    )


def test_ragas_evaluator_with_fallbacks_bleu_rouge_in_offline_tier() -> None:
    """BleuScore and RougeScore must be part of the offline metrics tier."""
    from src.evaluation.ragas_evaluator_with_fallbacks import RAGASEvaluatorWithFallbacks

    evaluator = RAGASEvaluatorWithFallbacks(config={})

    offline_class_names = {m.__class__.__name__ for m in evaluator.nonllm_metrics}
    assert "BleuScore" in offline_class_names or "RougeScore" in offline_class_names, (
        f"Expected BleuScore or RougeScore in nonllm_metrics, got: {offline_class_names}"
    )


# ---------------------------------------------------------------------------
# Contract-level: result_source and mock_data integrity
# ---------------------------------------------------------------------------

def test_result_source_never_contains_mock_for_offline_path(monkeypatch) -> None:
    """result_source strings for CI/offline paths must not contain 'mock'."""
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = object.__new__(RagasEvaluator)
    evaluator.ragas_available = True
    evaluator.metrics = []
    evaluator._prepare_evaluation_data = lambda testset, responses: {
        "question": ["Q1"],
        "answer": ["A1"],
        "contexts": [["C1"]],
        "ground_truth": ["G1"],
    }

    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.fix_ragas_dataset_format",
        lambda data: data,
    )
    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.create_safe_ragas_dataset",
        lambda data: type("_D", (), {
            "__len__": lambda self: 1,
            "column_names": ["question"],
        })(),
    )
    monkeypatch.setattr(
        "src.evaluation.ragas_evaluator.RagasModelDumpFix.safe_ragas_evaluate",
        lambda dataset, metrics: None,
    )

    result = evaluator.evaluate({"questions": ["Q1"]}, [{"answer": "A1"}])

    assert "mock" not in result["result_source"], (
        f"result_source must not contain 'mock' for offline path: {result['result_source']}"
    )
