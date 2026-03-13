from src.evaluation.ragas_evaluator import DomainRegexHeuristic, RagasEvaluator
from src.optimization.dpo_alignment import DirectPreferenceOptimizationPipeline


def test_domain_regex_heuristic_scores_required_terms_and_patterns():
    heuristic = DomainRegexHeuristic(
        required_terms=["report"],
        regex_patterns=[r"IMPORTANT:\s+.+"],
    )

    assert heuristic.score("This report contains IMPORTANT: action items") == 1.0
    assert heuristic.score("This report is incomplete") == 0.5


def test_weighted_domain_score_uses_configured_weights():
    evaluator = object.__new__(RagasEvaluator)
    evaluator.metric_weights = {
        "context_precision": 0.7,
        "faithfulness": 0.2,
        "domain_regex": 0.1,
    }

    score = evaluator._compute_weighted_domain_score(
        {
            "context_precision": {"mean": 0.8},
            "faithfulness": {"mean": 0.5},
            "domain_regex": {"mean": 1.0},
        }
    )

    assert round(score, 4) == 0.76


def test_evaluate_responses_merges_multimodal_metrics_and_alignment(tmp_path):
    evaluator = object.__new__(RagasEvaluator)
    evaluator.ragas_available = True
    evaluator.metric_weights = {
        "context_precision": 0.6,
        "faithfulness": 0.2,
        "domain_regex": 0.2,
    }
    evaluator.heuristic = DomainRegexHeuristic(required_terms=["robotic"])
    evaluator.multimodal_evaluator = __import__(
        "src.evaluation.multimodal_metrics", fromlist=["MultimodalResponseEvaluator"]
    ).MultimodalResponseEvaluator()
    evaluator.alignment_pipeline = DirectPreferenceOptimizationPipeline(
        {"output_dir": str(tmp_path), "auto_run_threshold": 1}
    )
    evaluator.ragas_metrics_config = {"alignment": {"confidence_threshold": 0.7}}
    evaluator.llm_roles = {}

    class _FakeResults:
        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame(
                {
                    "context_precision": [0.8],
                    "faithfulness": [0.7],
                }
            )

    evaluator._format_results = RagasEvaluator._format_results.__get__(evaluator, RagasEvaluator)
    evaluator._compute_weighted_domain_score = RagasEvaluator._compute_weighted_domain_score.__get__(evaluator, RagasEvaluator)
    evaluator._merge_metric_payloads = RagasEvaluator._merge_metric_payloads.__get__(evaluator, RagasEvaluator)
    evaluator._collect_alignment_failures = RagasEvaluator._collect_alignment_failures.__get__(evaluator, RagasEvaluator)
    evaluator._score_domain_regex = RagasEvaluator._score_domain_regex.__get__(evaluator, RagasEvaluator)

    base = evaluator._format_results(_FakeResults(), [1.0])
    merged = evaluator._merge_metric_payloads(
        base,
        evaluator.multimodal_evaluator.evaluate_responses(
            [
                {
                    "question": "What is shown?",
                    "answer": "A robotic assembly arm is visible.",
                    "contexts": [
                        {"type": "image", "content": "robotic assembly arm", "ocr_text": "robotic arm"}
                    ],
                    "reference": "A robotic assembly arm is visible.",
                    "confidence": 0.2,
                }
            ]
        )["metrics"],
    )
    alignment = evaluator._collect_alignment_failures(
        [
            {
                "question": "What is shown?",
                "answer": "",
                "reference": "A robotic assembly arm is visible.",
                "confidence": 0.2,
            }
        ]
    )

    assert "multimodal_faithfulness" in merged["metrics"]
    assert merged["summary"]["domain_score"] > 0
    assert alignment["queued_failures"] == 1
    assert alignment["training_run"]["executed"] is True