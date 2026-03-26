from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from services.testset.payloads import DraftSample


@dataclass
class DeduplicationResult:
    samples: List[DraftSample]
    dropped_duplicates: int
    duplicate_ratio: float


@dataclass(frozen=True)
class SampleDeduplicatorConfig:
    jaccard_threshold: float = 0.82
    shingle_size: int = 4
    max_duplicate_ratio: float = 0.25


class SampleDeduplicator:
    """Removes near-duplicate samples using shingled Jaccard similarity."""

    def __init__(self, config: SampleDeduplicatorConfig | None = None) -> None:
        self._config = config or SampleDeduplicatorConfig()

    @property
    def config(self) -> SampleDeduplicatorConfig:
        return self._config

    def apply(self, samples: Sequence[DraftSample]) -> DeduplicationResult:
        if not samples:
            return DeduplicationResult(samples=[], dropped_duplicates=0, duplicate_ratio=0.0)

        kept: List[DraftSample] = []
        signatures: List[Set[str]] = []
        dropped = 0

        for sample in samples:
            signature = _build_signature(sample, self._config.shingle_size)
            if _is_near_duplicate(signature, signatures, self._config.jaccard_threshold):
                dropped += 1
                continue
            signatures.append(signature)
            kept.append(sample)

        ratio = dropped / len(samples)
        return DeduplicationResult(samples=kept, dropped_duplicates=dropped, duplicate_ratio=ratio)


def _build_signature(sample: DraftSample, shingle_size: int) -> Set[str]:
    question = sample.question.lower().strip()
    answer = sample.answer.lower().strip()
    persona_id = sample.persona_id or ""
    scenario_id = sample.scenario_id or ""

    tokens = _tokenize(" ".join(filter(None, [question, answer, persona_id, scenario_id])))
    if not tokens:
        return {""}

    window = max(1, shingle_size)
    shingles: Set[str] = set()
    if len(tokens) < window:
        shingles.add(" ".join(tokens))
    else:
        for index in range(len(tokens) - window + 1):
            shingle = " ".join(tokens[index : index + window])
            shingles.add(shingle)
    return shingles or {""}


def _tokenize(text: str) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return cleaned.split(" ")


def _is_near_duplicate(candidate: Set[str], signatures: Iterable[Set[str]], threshold: float) -> bool:
    if not candidate:
        return False
    for existing in signatures:
        score = _jaccard(candidate, existing)
        if score >= threshold:
            return True
    return False


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


__all__ = [
    "SampleDeduplicator",
    "SampleDeduplicatorConfig",
    "DeduplicationResult",
]
