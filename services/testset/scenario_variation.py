from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from services.testset.payloads import SourceChunk


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    instructions: str


_DEFAULT_SCENARIOS: Sequence[ScenarioDefinition] = (
    ScenarioDefinition(
        scenario_id="baseline",
        label="Baseline Summary",
        instructions="Provide a factual and concise response covering {focus}.",
    ),
    ScenarioDefinition(
        scenario_id="troubleshooting",
        label="Troubleshooting Guidance",
        instructions="Outline remediation steps and highlight risks related to {focus}.",
    ),
    ScenarioDefinition(
        scenario_id="compliance",
        label="Compliance Checklist",
        instructions="List compliance duties and verification points for {focus}.",
    ),
    ScenarioDefinition(
        scenario_id="executive",
        label="Executive Brief",
        instructions="Summarize high-impact decisions and required actions referencing {focus}.",
    ),
)

_ADDITIONAL_MODIFIERS: Sequence[ScenarioDefinition] = (
    ScenarioDefinition(
        scenario_id="risk-assessment",
        label="Risk Assessment",
        instructions="Assess potential failure modes and mitigation steps around {focus}.",
    ),
    ScenarioDefinition(
        scenario_id="training",
        label="Training Guidance",
        instructions="Explain how to train new team members on procedures tied to {focus}.",
    ),
    ScenarioDefinition(
        scenario_id="audit",
        label="Audit Prep",
        instructions="Prepare audit questions and evidence expectations for {focus}.",
    ),
)


class ScenarioVariationGenerator:
    """Generates deterministic scenario definitions tailored to the provided chunks."""

    def build_definitions(
        self,
        *,
        chunks: Sequence[SourceChunk],
        requested_count: int,
        rng: random.Random,
    ) -> List[ScenarioDefinition]:
        focus = _determine_focus(chunks)
        target = max(3, requested_count)

        candidates: List[ScenarioDefinition] = [
            _render_template(template, focus) for template in _DEFAULT_SCENARIOS
        ]

        if len(candidates) < target:
            candidates.extend(_synthesise_additional(focus, target - len(candidates)))

        if len(candidates) < target:
            candidates.extend(
                _clone_with_suffix(candidates, focus, target - len(candidates))
            )

        rng.shuffle(candidates)
        return candidates[:target]


__all__ = ["ScenarioDefinition", "ScenarioVariationGenerator"]


def _determine_focus(chunks: Sequence[SourceChunk]) -> str:
    if not chunks:
        return "the provided material"

    document_ids = {chunk.document_id for chunk in chunks if chunk.document_id}
    sections = _collect_metadata(chunks, "section")
    topics = _collect_metadata(chunks, "topic")

    focus_parts: List[str] = []
    if topics:
        focus_parts.extend(sorted(topics)[:2])
    if sections:
        focus_parts.extend(sorted(sections)[:2])
    if not focus_parts and document_ids:
        focus_parts.append(next(iter(sorted(document_ids))))
    return " / ".join(focus_parts) or "the provided material"


def _collect_metadata(chunks: Sequence[SourceChunk], key: str) -> List[str]:
    values: List[str] = []
    for chunk in chunks:
        metadata_value = chunk.metadata.get(key) if chunk.metadata else None
        if not metadata_value:
            continue
        if isinstance(metadata_value, (list, tuple, set)):
            values.extend(str(item).strip() for item in metadata_value if item)
        else:
            values.append(str(metadata_value).strip())
    seen = set()
    ordered: List[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(value)
    return ordered


def _render_template(template: ScenarioDefinition, focus: str) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id=template.scenario_id,
        label=template.label,
        instructions=template.instructions.format(focus=focus),
    )


def _synthesise_additional(focus: str, count: int) -> List[ScenarioDefinition]:
    pool = list(_ADDITIONAL_MODIFIERS)
    result: List[ScenarioDefinition] = []
    index = 0
    while len(result) < count and pool:
        template = pool[index % len(pool)]
        scenario_id = f"{template.scenario_id}-{index + 1}"
        result.append(
            ScenarioDefinition(
                scenario_id=scenario_id,
                label=template.label,
                instructions=template.instructions.format(focus=focus),
            )
        )
        index += 1
    return result


def _clone_with_suffix(
    definitions: Iterable[ScenarioDefinition],
    focus: str,
    count: int,
) -> List[ScenarioDefinition]:
    clones: List[ScenarioDefinition] = []
    iterator = list(definitions)
    idx = 0
    while len(clones) < count and iterator:
        base = iterator[idx % len(iterator)]
        scenario_id = f"{base.scenario_id}-alt-{idx + 1}"
        clones.append(
            ScenarioDefinition(
                scenario_id=scenario_id,
                label=f"{base.label} (Alt {idx + 1})",
                instructions=f"{base.instructions} Emphasize {focus} details.",
            )
        )
        idx += 1
    return clones
