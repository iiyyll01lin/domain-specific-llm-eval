from src.evaluation.ragas_evaluator import DomainRegexHeuristic, RagasEvaluator


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