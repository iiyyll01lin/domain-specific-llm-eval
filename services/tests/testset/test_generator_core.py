from __future__ import annotations

from typing import List

from services.testset.generator_core import GeneratorCore
from services.testset.payloads import SourceChunk


def _build_chunks() -> List[SourceChunk]:
    return [
        SourceChunk(
            chunk_id=f"chunk-{index}",
            document_id="doc-alpha" if index < 2 else "doc-beta",
            text="This policy outlines safety procedures for inspectors and field engineers.",
            metadata={"section": f"1.{index}"},
        )
        for index in range(4)
    ]


def test_generator_is_deterministic_for_same_seed() -> None:
    chunks = _build_chunks()
    config = {
        "method": "ragas",
        "max_total_samples": 5,
        "samples_per_document": 2,
        "seed": 123,
        "selected_strategies": ["baseline", "drilldown"],
    }

    generator = GeneratorCore()
    samples_a, stats_a, meta_a = generator.generate(chunks=chunks, config=config)
    samples_b, stats_b, meta_b = generator.generate(chunks=chunks, config=config)

    assert [sample.eval_sample.user_input for sample in samples_a] == [
        sample.eval_sample.user_input for sample in samples_b
    ][: len(samples_a)]
    assert [sample.eval_sample.reference for sample in samples_a] == [
        sample.eval_sample.reference for sample in samples_b
    ][: len(samples_a)]
    assert stats_a == stats_b
    assert meta_a == meta_b


def test_generator_respects_max_total_samples() -> None:
    chunks = _build_chunks()
    config = {
        "method": "ragas",
        "max_total_samples": 3,
        "samples_per_document": 5,
        "seed": 11,
    }

    generator = GeneratorCore()
    samples, stats, _ = generator.generate(chunks=chunks, config=config)
    assert len(samples) == 3
    assert stats.filtered == 3


def test_generator_applies_persona_and_scenarios() -> None:
    chunks = _build_chunks()
    config = {
        "method": "ragas",
        "max_total_samples": 4,
        "seed": 21,
        "persona_profile": {
            "id": "qa-analyst",
            "name": "QA Analyst",
            "role": "Quality Review",
            "description": "Focus on edge cases and ensure clear risk call-outs.",
        },
    }

    generator = GeneratorCore()
    samples, _, metadata = generator.generate(chunks=chunks, config=config)

    assert samples, "expected generated samples"
    sample_question = samples[0].eval_sample.user_input or ""
    assert "Persona context" in sample_question
    assert "Scenario:" in sample_question
    assert metadata["persona"]["id"] == "qa-analyst"
