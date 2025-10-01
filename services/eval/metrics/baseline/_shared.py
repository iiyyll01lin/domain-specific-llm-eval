from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _extract_tokens(*segments: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for segment in segments:
        for raw in segment:
            for match in _TOKEN_PATTERN.finditer(raw.lower()):
                tokens.append(match.group())
    return tokens


def _token_sets(texts: Sequence[str]) -> set[str]:
    if not texts:
        return set()
    return set(_extract_tokens(texts))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def _overlap_ratio(left: Sequence[str], right: Sequence[str]) -> tuple[float, int, int]:
    left_tokens = _token_sets(list(left))
    right_tokens = _token_sets(list(right))
    if not left_tokens or not right_tokens:
        return 0.0, 0, len(left_tokens)
    matched = len(left_tokens & right_tokens)
    score = _safe_ratio(matched, len(left_tokens))
    return score, matched, len(left_tokens)


def _context_precision(contexts: Sequence[str], answer: Sequence[str]) -> tuple[float, int, int]:
    context_tokens = _token_sets(list(contexts))
    answer_tokens = _token_sets(list(answer))
    if not context_tokens or not answer_tokens:
        return 0.0, 0, len(context_tokens)
    matched = len(context_tokens & answer_tokens)
    score = _safe_ratio(matched, len(context_tokens))
    return score, matched, len(context_tokens)


def _round(value: float) -> float:
    return 0.0 if math.isnan(value) else round(value, 6)
