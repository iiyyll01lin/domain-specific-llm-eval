from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Sequence

from services.testset.payloads import DraftSample


@dataclass
class QualityFilterResult:
    samples: List[DraftSample]
    dropped_duplicates: int
    dropped_length: int


@dataclass(frozen=True)
class QualityFilterConfig:
    min_question_chars: int = 12
    max_question_chars: int = 360
    min_answer_chars: int = 8
    similarity_threshold: float = 0.92


class QualityFilter:
    """Filters out low quality or near-duplicate samples before persistence."""

    def __init__(self, config: QualityFilterConfig | None = None) -> None:
        self._config = config or QualityFilterConfig()

    def apply(
        self,
        samples: Sequence[DraftSample],
        *,
        rng: random.Random,
    ) -> QualityFilterResult:
        del rng

        kept: List[DraftSample] = []
        dropped_duplicates = 0
        dropped_length = 0
        fingerprints: List[str] = []

        for sample in samples:
            question = sample.question.strip()
            answer = sample.answer.strip()
            if not _passes_length_checks(question, answer, self._config):
                dropped_length += 1
                continue

            fingerprint = _normalise(question)
            if _is_duplicate(fingerprint, fingerprints, self._config.similarity_threshold):
                dropped_duplicates += 1
                continue

            fingerprints.append(fingerprint)
            kept.append(sample)

        return QualityFilterResult(
            samples=kept,
            dropped_duplicates=dropped_duplicates,
            dropped_length=dropped_length,
        )


def _passes_length_checks(question: str, answer: str, config: QualityFilterConfig) -> bool:
    return (
        config.min_question_chars <= len(question) <= config.max_question_chars
        and len(answer) >= config.min_answer_chars
    )


def _normalise(text: str) -> str:
    lowered = text.lower()
    return " ".join(lowered.split())


def _is_duplicate(candidate: str, fingerprints: Sequence[str], threshold: float) -> bool:
    if candidate in fingerprints:
        return True
    for existing in fingerprints:
        if SequenceMatcher(None, candidate, existing).ratio() >= threshold:
            return True
    return False


__all__ = ["QualityFilter", "QualityFilterConfig", "QualityFilterResult"]
