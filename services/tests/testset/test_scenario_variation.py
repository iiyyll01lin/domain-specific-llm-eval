from __future__ import annotations

import random

from services.testset.payloads import SourceChunk
from services.testset.scenario_variation import ScenarioVariationGenerator


def _build_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="chunk-1",
            document_id="doc-operations",
            text="This section details inspection cadence and escalation paths.",
            metadata={"section": "Inspection", "topic": ["Safety", "Escalation"]},
        ),
        SourceChunk(
            chunk_id="chunk-2",
            document_id="doc-operations",
            text="Include reporting obligations to regional managers.",
            metadata={"section": "Reporting"},
        ),
    ]


def test_scenario_variations_reach_minimum_threshold() -> None:
    generator = ScenarioVariationGenerator()
    rng = random.Random(42)

    definitions = generator.build_definitions(
        chunks=_build_chunks(),
        requested_count=2,
        rng=rng,
    )

    assert len(definitions) >= 3
    scenario_ids = {definition.scenario_id for definition in definitions}
    assert len(scenario_ids) == len(definitions)


def test_scenario_variations_are_deterministic() -> None:
    generator = ScenarioVariationGenerator()
    chunks = _build_chunks()

    first = generator.build_definitions(chunks=chunks, requested_count=5, rng=random.Random(7))
    second = generator.build_definitions(chunks=chunks, requested_count=5, rng=random.Random(7))

    assert [definition.scenario_id for definition in first] == [
        definition.scenario_id for definition in second
    ]
    assert [definition.instructions for definition in first] == [
        definition.instructions for definition in second
    ]
    assert any("Safety" in definition.instructions for definition in first)
