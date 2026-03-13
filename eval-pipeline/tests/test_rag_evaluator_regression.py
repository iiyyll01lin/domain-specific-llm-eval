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