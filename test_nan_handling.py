from __future__ import annotations

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent / "eval-pipeline" / "src"))


def test_nan_handling() -> None:
    from utils.nan_handling import is_valid_score, safe_mean, safe_std

    test_scores = [0.5, 0.8, float("nan"), 0.3, None, float("inf"), 0.9]

    assert round(safe_mean(test_scores), 4) == 0.625
    assert round(safe_std(test_scores), 4) > 0
    assert is_valid_score(0.5) is True
    assert is_valid_score(float("nan")) is False
    assert is_valid_score(float("inf")) is False


def test_evaluator_imports() -> None:
    import evaluation.comprehensive_rag_evaluator  # noqa: F401
    import evaluation.ragas_evaluator_fixed  # noqa: F401
