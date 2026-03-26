from __future__ import annotations

import random

from services.testset.payloads import DraftSample
from services.testset.pre_filter import QualityFilter, QualityFilterConfig


def _make_sample(question: str, answer: str = "Standard operating procedure.") -> DraftSample:
    return DraftSample(
        question=question,
        answer=answer,
        reference_chunk_ids=["chunk-1"],
        reference_contexts=["context"],
        strategy="baseline",
        metadata={"document_id": "doc-1"},
    )


def test_quality_filter_removes_majority_duplicates() -> None:
    duplicates = [_make_sample("Describe the lockout tagout safety protocol." ) for _ in range(10)]
    unique = _make_sample("Summarize the emergency evacuation process.")
    samples = duplicates + [unique]

    flt = QualityFilter(QualityFilterConfig(similarity_threshold=0.9))
    result = flt.apply(samples, rng=random.Random(5))

    assert unique.question in [sample.question for sample in result.samples]
    duplicate_removal_ratio = result.dropped_duplicates / len(duplicates)
    assert duplicate_removal_ratio >= 0.9


def test_quality_filter_enforces_length_constraints() -> None:
    too_short_question = _make_sample("Short?", "Valid answer")
    adequate = _make_sample("Provide the audit logging procedure for nightly jobs.")

    flt = QualityFilter()
    result = flt.apply([too_short_question, adequate], rng=random.Random(2))

    questions = [sample.question for sample in result.samples]
    assert too_short_question.question not in questions
    assert result.dropped_length == 1
