from __future__ import annotations

import pytest

from services.testset.dedupe import SampleDeduplicator, SampleDeduplicatorConfig
from services.testset.payloads import DraftSample


def _build_sample(question: str, *, persona_id: str | None = None, scenario_id: str | None = None) -> DraftSample:
    return DraftSample(
        question=question,
        answer="The policy mandates annual review.",
        reference_chunk_ids=["chunk-1"],
        reference_contexts=["context"],
        strategy="baseline",
        persona_id=persona_id,
        scenario_id=scenario_id,
    )


def test_deduplicator_drops_near_duplicates() -> None:
    samples = [
        _build_sample("Alpha beta gamma guidance"),
        _build_sample("alpha beta gamma guidance"),
        _build_sample("Distinct instructions for auditors"),
    ]

    deduplicator = SampleDeduplicator(SampleDeduplicatorConfig(jaccard_threshold=0.75, shingle_size=2))
    result = deduplicator.apply(samples)

    assert len(result.samples) == 2
    assert result.dropped_duplicates == 1
    assert result.duplicate_ratio == pytest.approx(1 / 3, rel=1e-2)


def test_deduplicator_retains_persona_variations() -> None:
    samples = [
        _build_sample("Summarize risk controls", persona_id="persona-a"),
        _build_sample("Summarize risk controls", persona_id="persona-b"),
    ]

    deduplicator = SampleDeduplicator()
    result = deduplicator.apply(samples)

    assert len(result.samples) == 2
    assert result.dropped_duplicates == 0
    assert result.duplicate_ratio == 0
