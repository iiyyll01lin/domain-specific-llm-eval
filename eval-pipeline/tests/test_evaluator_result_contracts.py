import sys
import types

import pandas as pd


def test_gates_result_exposes_normalized_contract_fields() -> None:
    from src.evaluation.gates_system import GatesSystem  # lazy
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


def test_contextual_keyword_evaluator_aggregate_result_includes_contract_fields(monkeypatch) -> None:
    from src.evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator  # lazy
    evaluator = object.__new__(ContextualKeywordEvaluator)
    evaluator.evaluate_batch = lambda evaluations: {
        "aggregate_stats": {
            "passed_evaluations": 1,
            "failed_evaluations": 0,
            "total_evaluations": 1,
            "mean_total_score": 0.8,
            "pass_rate": 1.0,
        },
        "individual_results": [
            {"total_score": 0.8, "keyword_relevance_score": 0.75}
        ],
        "evaluation_config": {"threshold": 0.6},
    }

    result = evaluator.evaluate_responses(
        [
            {
                "answer": "A robotic assembly arm is visible.",
                "expected_keywords": ["robotic", "assembly"],
            }
        ]
    )

    assert result["success"] is True
    assert result["result_source"] == "contextual_keyword_evaluator"
    assert result["mock_data"] is False
    assert result["contract_version"]


def test_rag_evaluator_success_result_includes_contract_fields(tmp_path) -> None:
    from src.evaluation.rag_evaluator import RAGEvaluator  # lazy
    evaluator = object.__new__(RAGEvaluator)
    testset_path = tmp_path / "testset.csv"
    testset_path.write_text("question,ground_truth\nWhat changed?,Answer\n", encoding="utf-8")

    evaluator.load_testset = lambda path: [{"question": "What changed?", "ground_truth": "Answer"}]  # type: ignore[method-assign]
    evaluator.query_rag_system = lambda question: {"rag_answer": "normalized answer"}  # type: ignore[method-assign]
    evaluator.keyword_evaluator = None
    evaluator.ragas_evaluator = None
    evaluator.extract_keywords_from_response = lambda answer: ["normalized"]  # type: ignore[method-assign]

    result = RAGEvaluator.evaluate_single_testset(evaluator, str(testset_path))

    assert result["success"] is True
    assert result["result_source"] == "rag_evaluator"
    assert result["mock_data"] is False
    assert result["contract_version"]


def test_comprehensive_rag_evaluator_missing_columns_result_includes_contract_fields(tmp_path) -> None:
    from src.evaluation.comprehensive_rag_evaluator_fixed import ComprehensiveRAGEvaluatorFixed  # lazy: avoids torch load at collection
    evaluator = object.__new__(ComprehensiveRAGEvaluatorFixed)
    evaluator.keyword_evaluator = None
    evaluator.ragas_evaluator = None
    evaluator.gates_system = None

    testset_path = tmp_path / "invalid.csv"
    pd.DataFrame({"question": ["What changed?"]}).to_csv(testset_path, index=False)

    result = ComprehensiveRAGEvaluatorFixed.evaluate_testset(
        evaluator,
        testset_path,
        tmp_path / "outputs",
    )

    assert result["success"] is False
    assert result["result_source"] == "comprehensive_rag_evaluator_fixed"
    assert result["error_stage"] == "validate_testset"
    assert result["contract_version"]


def test_comprehensive_rag_evaluator_contextual_result_includes_contract_fields() -> None:
    from src.evaluation.comprehensive_rag_evaluator import ComprehensiveRAGEvaluator  # lazy
    evaluator = object.__new__(ComprehensiveRAGEvaluator)
    evaluator.contextual_functions = {
        "weighted_keyword_score": lambda mandatory, answer, weights, optional: (0.8, 0.75, 0.2, [answer]),
    }
    evaluator.contextual_weights = {"mandatory": 0.8, "optional": 0.2}
    evaluator.contextual_threshold = 0.6

    result = ComprehensiveRAGEvaluator.evaluate_contextual_keywords(
        evaluator,
        "A robotic assembly arm is visible.",
        ["robotic", "assembly"],
    )

    assert result["success"] is True
    assert result["result_source"] == "comprehensive_rag_evaluator"
    assert result["contract_version"]


def test_pipeline_orchestrator_stage_result_includes_contract_fields(tmp_path) -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: avoids importlib hang at collection
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.run_id = "run-test"
    orchestrator.output_dirs = {
        "metadata": tmp_path / "metadata",
        "testsets": tmp_path / "testsets",
        "evaluations": tmp_path / "evaluations",
        "reports": tmp_path / "reports",
    }
    for path in orchestrator.output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    class _MemoryTracker:
        def log_memory_usage(self, _stage):
            return None

    orchestrator.memory_tracker = _MemoryTracker()
    orchestrator._run_taxonomy_discovery = lambda: None
    orchestrator._install_requested_runbooks = lambda: []

    orchestrator.document_processor = type(
        "_Processor",
        (),
        {
            "process_documents": lambda self: [
                {"source_file": "doc1.txt", "content": "text"}
            ]
        },
    )()
    orchestrator.testset_generator = type(
        "_Generator",
        (),
        {
            "generate_comprehensive_testset": lambda self, document_paths, output_dir: {
                "testset": [{"question": "Q1", "answer": "A1"}],
                "metadata": {"generation_method": "mock"},
                "results_by_method": {},
            }
        },
    )()

    result = PipelineOrchestrator._run_testset_generation(orchestrator)

    assert result["success"] is True
    assert result["result_source"] == "pipeline_orchestrator_testset_generation"
    assert result["contract_version"]