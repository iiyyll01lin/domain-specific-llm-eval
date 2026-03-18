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