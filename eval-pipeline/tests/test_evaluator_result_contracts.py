import sys
import types

from src.evaluation.gates_system import GatesSystem


def test_gates_result_exposes_normalized_contract_fields() -> None:
    system = GatesSystem({"evaluation": {"gates": {}}})
    result = system.evaluate_gates(
        {
            "contextual_keyword": {"success": False, "error": "missing"},
            "ragas": {"success": False, "error": "missing"},
        }
    )

    payload = result.as_dict()
    assert payload["result_source"] == "gates_system"
    assert payload["mock_data"] is False
    assert payload["contract_version"]


def test_hybrid_evaluator_result_includes_contract_fields(monkeypatch) -> None:
    fake_contextual_module = types.SimpleNamespace(
        ContextualKeywordGate=object,
        get_contextual_segments=lambda text: [text],
        weighted_keyword_score=lambda auto_keywords, rag_answer, keyword_weights: (0.9, 0.7, 0.2, auto_keywords),
    )
    fake_feedback_module = types.SimpleNamespace(
        adaptive_exponential_smoothing=lambda values, alpha=0.3: values,
        calculate_adaptive_window_size=lambda *args, **kwargs: 3,
        calculate_feedback_consistency=lambda *args, **kwargs: 1.0,
        needs_human_feedback_dynamic=lambda *args, **kwargs: False,
    )
    fake_ragas_module = types.SimpleNamespace(
        evaluate=lambda *args, **kwargs: {},
        metrics=types.SimpleNamespace(
            answer_correctness=object(),
            answer_relevancy=object(),
            answer_similarity=object(),
            context_precision=object(),
            context_recall=object(),
            faithfulness=object(),
        ),
    )
    monkeypatch.setitem(sys.modules, "contextual_keyword_gate", fake_contextual_module)
    monkeypatch.setitem(sys.modules, "dynamic_ragas_gate_with_human_feedback", fake_feedback_module)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_ragas_module.metrics)

    from src.evaluation.hybrid_evaluator import HybridEvaluator

    evaluator = HybridEvaluator({"evaluation": {}})
    monkeypatch.setattr(
        evaluator,
        "_evaluate_contextual_keywords",
        lambda *args, **kwargs: {"contextual_total_score": 0.9, "contextual_keyword_pass": True},
    )
    monkeypatch.setattr(
        evaluator,
        "_evaluate_with_ragas",
        lambda *args, **kwargs: {"ragas_composite_score": 0.8, "ragas_pass": True},
    )
    monkeypatch.setattr(
        evaluator,
        "_evaluate_semantic_similarity",
        lambda *args, **kwargs: {"semantic_similarity": 0.85, "semantic_pass": True},
    )
    monkeypatch.setattr(
        evaluator,
        "_assess_human_feedback_need",
        lambda *args, **kwargs: {"needs_human_feedback": False},
    )
    monkeypatch.setattr(evaluator, "_calculate_overall_score", lambda *args, **kwargs: 0.85)
    monkeypatch.setattr(evaluator, "_determine_overall_pass", lambda *args, **kwargs: True)

    result = evaluator.evaluate_rag_response(
        question="Q1",
        rag_answer="A1",
        expected_answer="A1",
        auto_keywords=["kw"],
        contexts=["ctx"],
    )

    assert result["result_source"] == "hybrid_evaluator"
    assert result["mock_data"] is False
    assert result["contract_version"]