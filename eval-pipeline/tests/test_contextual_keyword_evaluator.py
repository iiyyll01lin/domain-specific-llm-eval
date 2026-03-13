from __future__ import annotations

from evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator


def _build_evaluator(monkeypatch) -> ContextualKeywordEvaluator:
    monkeypatch.setattr(
        "evaluation.contextual_keyword_evaluator.OFFLINE_MANAGER_AVAILABLE", False
    )
    monkeypatch.setattr(
        "evaluation.contextual_keyword_evaluator.SENTENCE_TRANSFORMERS_AVAILABLE",
        False,
    )
    monkeypatch.setattr(
        "evaluation.contextual_keyword_evaluator.SPACY_AVAILABLE", False
    )
    return ContextualKeywordEvaluator(
        {
            "threshold": 0.6,
            "weights": {"mandatory": 0.8, "optional": 0.2},
            "keyword_synonyms": {
                "thickness gauge": ["caliper"],
                "surface inspection": ["surface check"],
            },
            "partial_match_weight": 0.8,
        }
    )


def test_fallback_scoring_handles_exact_synonym_and_partial_matches(monkeypatch) -> None:
    evaluator = _build_evaluator(monkeypatch)

    result = evaluator.evaluate_response(
        "Use a caliper during the surface inspect procedure to validate the panel.",
        ["thickness gauge", "surface inspection", "panel validation"],
    )

    assert result["passes_threshold"] is True
    assert result["keyword_relevance_score"] >= 0.6
    assert "thickness gauge" in result["matched_mandatory"]
    assert any(
        detail["match_type"] in {"synonym", "partial"}
        for detail in result["mandatory_match_details"]
        if detail["keyword"] in {"thickness gauge", "surface inspection"}
    )


def test_evaluate_responses_uses_normalized_rag_fields(monkeypatch) -> None:
    evaluator = _build_evaluator(monkeypatch)

    result = evaluator.evaluate_responses(
        [
            {
                "rag_answer": "A caliper is used for thickness checks.",
                "extracted_keywords": ["thickness gauge"],
            },
            {
                "answer": "Surface check is required before release.",
                "mandatory_keywords": ["surface inspection"],
            },
        ]
    )

    assert result["total_evaluations"] == 2
    assert result["pass_count"] >= 1
    assert "contextual_keyword_score" in result["metrics"]
    assert len(result["metrics"]["keyword_relevance_score"]["scores"]) == 2


def test_evaluate_testset_merges_expected_and_optional_keywords(monkeypatch) -> None:
    evaluator = _build_evaluator(monkeypatch)

    results = evaluator.evaluate_testset(
        {
            "qa_pairs": [
                {
                    "reference": "Use a caliper and complete surface check before shipment.",
                    "expected_keywords": ["thickness gauge"],
                    "mandatory_keywords": ["surface inspection"],
                    "optional_keywords": ["shipment"],
                }
            ]
        }
    )

    assert len(results) == 1
    assert results[0]["passed"] is True
    assert results[0]["keyword_evaluation"]["keyword_relevance_score"] >= 0.6
