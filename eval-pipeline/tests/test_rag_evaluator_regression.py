from __future__ import annotations

from pathlib import Path

from src.evaluation.rag_evaluator import RAGEvaluator
from src.optimization.dpo_alignment import DirectPreferenceOptimizationPipeline


def test_evaluate_testset_uses_normalized_rag_response_fields() -> None:
    evaluator = object.__new__(RAGEvaluator)

    def _query(_question: str):
        return {
            "rag_answer": "normalized answer",
            "rag_confidence": 0.9,
            "rag_contexts": [{"type": "text", "content": "ctx"}],
            "response_time": 0.12,
        }

    evaluator.query_rag_system = _query  # type: ignore[method-assign]

    results = RAGEvaluator.evaluate_testset(
        evaluator,
        {"qa_pairs": [{"user_input": "What changed?", "reference": "answer"}]},
    )

    assert results[0]["rag_answer"] == "normalized answer"
    assert results[0]["rag_confidence"] == 0.9
    assert results[0]["rag_contexts"][0]["content"] == "ctx"


def test_dpo_alignment_persists_and_reloads_queue(tmp_path: Path) -> None:
    pipeline = DirectPreferenceOptimizationPipeline({"output_dir": str(tmp_path)})
    pipeline.ingest_failure("Q", "bad", "good", metadata={"confidence": 0.1})

    restored = DirectPreferenceOptimizationPipeline({"output_dir": str(tmp_path)})

    assert len(restored.failure_queue) == 1
    assert restored.failure_queue[0]["metadata"]["confidence"] == 0.1


def test_evaluate_single_testset_surfaces_keyword_evaluator_errors(tmp_path: Path) -> None:
    evaluator = object.__new__(RAGEvaluator)
    testset_path = tmp_path / "testset.csv"
    testset_path.write_text("question,ground_truth\nWhat changed?,Answer\n", encoding="utf-8")

    evaluator.load_testset = lambda path: [{"question": "What changed?", "ground_truth": "Answer"}]  # type: ignore[method-assign]
    evaluator.query_rag_system = lambda question: {"rag_answer": "normalized answer"}  # type: ignore[method-assign]

    class _BrokenKeywordEvaluator:
        def evaluate_responses(self, _responses):
            raise RuntimeError("keyword evaluator exploded")

    evaluator.keyword_evaluator = _BrokenKeywordEvaluator()
    evaluator.ragas_evaluator = None
    evaluator.extract_keywords_from_response = lambda answer: ["normalized"]  # type: ignore[method-assign]

    result = RAGEvaluator.evaluate_single_testset(evaluator, str(testset_path))

    assert result["error"] == "keyword evaluator exploded"
    assert result["keyword_metrics"]["error"] == "keyword evaluator exploded"
    assert result["result_source"] == "rag_evaluator"
    assert result["error_stage"] == "keyword_evaluation"
    assert result["contract_version"]


def test_evaluate_single_testset_honors_contract_based_keyword_failures(tmp_path: Path) -> None:
    evaluator = object.__new__(RAGEvaluator)
    testset_path = tmp_path / "testset.csv"
    testset_path.write_text("question,ground_truth\nWhat changed?,Answer\n", encoding="utf-8")

    evaluator.load_testset = lambda path: [{"question": "What changed?", "ground_truth": "Answer"}]  # type: ignore[method-assign]
    evaluator.query_rag_system = lambda question: {"rag_answer": "normalized answer"}  # type: ignore[method-assign]
    evaluator.extract_keywords_from_response = lambda answer: ["normalized"]  # type: ignore[method-assign]

    class _ContractBrokenKeywordEvaluator:
        def evaluate_responses(self, _responses):
            return {
                "success": False,
                "error": "contract failure",
                "result_source": "keyword_evaluator",
                "error_stage": "evaluate_responses",
            }

    evaluator.keyword_evaluator = _ContractBrokenKeywordEvaluator()
    evaluator.ragas_evaluator = None

    result = RAGEvaluator.evaluate_single_testset(evaluator, str(testset_path))

    assert result["success"] is False
    assert result["error"] == "contract failure"
    assert result["keyword_metrics"]["success"] is False
    assert result["error_stage"] == "keyword_evaluation"


def test_evaluate_testsets_separates_failed_results(tmp_path: Path) -> None:
    evaluator = object.__new__(RAGEvaluator)
    evaluator.evaluate_single_testset = lambda path: {"testset_path": path, "error": "failed"} if "bad" in path else {"testset_path": path, "total_questions": 2, "successful_queries": 2}  # type: ignore[method-assign]

    result = RAGEvaluator.evaluate_testsets(
        evaluator,
        [str(tmp_path / "good.csv"), str(tmp_path / "bad.csv")],
    )

    assert result["failed_testsets"] == 1
    assert len(result["failed_results"]) == 1
    assert result["total_queries"] == 2
    assert result["result_source"] == "rag_evaluator_batch"
    assert result["error_stage"] == "partial_failure"


def test_evaluate_single_testset_rejects_dataframe_argument() -> None:
    """evaluate_single_testset() must raise TypeError immediately when given a
    DataFrame (or any non-str/Path value) to prevent silent misuse."""
    import pandas as pd
    import pytest

    evaluator = object.__new__(RAGEvaluator)

    with pytest.raises(TypeError) as exc_info:
        RAGEvaluator.evaluate_single_testset(evaluator, pd.DataFrame())  # type: ignore[arg-type]

    assert "DataFrame" in str(exc_info.value) or "expected" in str(exc_info.value).lower()
    assert "evaluate_single_testset" in str(exc_info.value)


def test_evaluate_single_testset_rejects_dict_argument() -> None:
    """evaluate_single_testset() must raise TypeError when given a dict."""
    import pytest

    evaluator = object.__new__(RAGEvaluator)

    with pytest.raises(TypeError):
        RAGEvaluator.evaluate_single_testset(evaluator, {"questions": ["Q1"]})  # type: ignore[arg-type]


def test_evaluate_single_testset_accepts_path_object(tmp_path: Path) -> None:
    """evaluate_single_testset() must accept a pathlib.Path without raising TypeError."""
    fake_xlsx = tmp_path / "testset.xlsx"
    fake_xlsx.write_bytes(b"fake")

    evaluator = object.__new__(RAGEvaluator)
    evaluator.keyword_evaluator = None
    evaluator.ragas_evaluator = None
    # load_testset will fail on the fake file, but TypeError must NOT be raised
    evaluator.load_testset = lambda path: (_ for _ in ()).throw(  # type: ignore[method-assign]
        FileNotFoundError("intentional")
    )

    result = RAGEvaluator.evaluate_single_testset(evaluator, fake_xlsx)
    # Should get an error result (FileNotFoundError), not a TypeError
    assert "error" in result or result.get("success") is False
    assert result["contract_version"]