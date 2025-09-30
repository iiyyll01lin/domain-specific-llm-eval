from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import random

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers.testset_schema import TestsetSample

from services.testset.dedupe import SampleDeduplicator
from services.testset.payloads import DraftSample, SourceChunk
from services.testset.persona_injector import PersonaInjectionResult, PersonaInjector
from services.testset.pre_filter import QualityFilter, QualityFilterResult
from services.testset.scenario_variation import (
    ScenarioDefinition,
    ScenarioVariationGenerator,
)


_DEFAULT_TEMPLATES: Sequence[str] = (
    "Summarize the following content: {snippet}",
    "What are the key actions described in: {snippet}",
    "Explain the primary objective discussed in: {snippet}",
    "List the critical compliance requirements mentioned in: {snippet}",
)


@dataclass(frozen=True)
class GenerationParameters:
    method: str
    max_total_samples: int
    seed: int
    samples_per_document: Optional[int] = None
    selected_strategies: Sequence[str] = field(default_factory=tuple)
    persona_profile: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GenerationParameters":
        method = str(payload.get("method", "unknown"))
        max_total_samples = int(payload.get("max_total_samples", 1))
        seed = int(payload.get("seed", 0))
        samples_per_document = payload.get("samples_per_document")
        if samples_per_document is not None:
            samples_per_document = int(samples_per_document)
        selected = payload.get("selected_strategies") or []
        if isinstance(selected, str):
            selected = [selected]
        persona_profile = payload.get("persona_profile")
        return cls(
            method=method,
            max_total_samples=max_total_samples,
            seed=seed,
            samples_per_document=samples_per_document,
            selected_strategies=tuple(selected),
            persona_profile=persona_profile if isinstance(persona_profile, Mapping) else None,
        )


@dataclass
class GenerationStats:
    generated: int
    filtered: int
    dropped_duplicates: int
    dropped_length: int


class GeneratorCore:
    """Deterministic question/answer generator orchestrating helper modules."""

    def __init__(
        self,
        *,
        templates: Sequence[str] | None = None,
        persona_injector: PersonaInjector | None = None,
        deduplicator: SampleDeduplicator | None = None,
        scenario_generator: ScenarioVariationGenerator | None = None,
        quality_filter: QualityFilter | None = None,
    ) -> None:
        self._templates = templates or _DEFAULT_TEMPLATES
        self._persona_injector = persona_injector or PersonaInjector()
        self._deduplicator = deduplicator or SampleDeduplicator()
        self._scenario_generator = scenario_generator or ScenarioVariationGenerator()
        self._quality_filter = quality_filter or QualityFilter()

    def generate(
        self,
        *,
        chunks: Sequence[SourceChunk],
        config: Mapping[str, Any],
    ) -> tuple[List[TestsetSample], GenerationStats, Dict[str, Any]]:
        params = GenerationParameters.from_mapping(config)
        rng = random.Random(params.seed)

        draft_samples = self._create_drafts(chunks=chunks, params=params, rng=rng)
        scenarios = self._scenario_generator.build_definitions(
            chunks=chunks,
            requested_count=max(3, len(draft_samples)) if draft_samples else 0,
            rng=rng,
        )
        processed = self._apply_personas_and_scenarios(
            drafts=draft_samples,
            persona_profile=params.persona_profile,
            scenarios=scenarios,
            rng=rng,
        )
        dedupe_result = self._deduplicator.apply(processed)
        filtered = self._quality_filter.apply(dedupe_result.samples, rng=rng)
        limited = filtered.samples[: params.max_total_samples]
        samples = [self._to_testset_sample(sample, params.method) for sample in limited]

        stats = GenerationStats(
            generated=len(draft_samples),
            filtered=len(limited),
            dropped_duplicates=dedupe_result.dropped_duplicates + filtered.dropped_duplicates,
            dropped_length=filtered.dropped_length,
        )
        metadata = {
            "seed": params.seed,
            "persona": params.persona_profile or None,
            "persona_count": 1 if params.persona_profile else 0,
            "scenarios": [
                {
                    "scenario_id": scenario.scenario_id,
                    "label": scenario.label,
                    "instructions": scenario.instructions,
                }
                for scenario in scenarios
            ],
            "scenario_count": len(scenarios),
            "strategies": list(params.selected_strategies),
            "deduplicated_count": len(dedupe_result.samples),
            "duplicate_ratio": dedupe_result.duplicate_ratio,
            "duplicate_ratio_limit": self._deduplicator.config.max_duplicate_ratio,
        }
        return samples, stats, metadata

    def _create_drafts(
        self,
        *,
        chunks: Sequence[SourceChunk],
        params: GenerationParameters,
        rng: random.Random,
    ) -> List[DraftSample]:
        if not chunks:
            return []

        per_document = params.samples_per_document or max(1, params.max_total_samples // len(chunks) or 1)
        strategies = list(params.selected_strategies) or ["baseline"]
        drafts: List[DraftSample] = []

        for chunk in chunks:
            for _ in range(per_document):
                if len(drafts) >= params.max_total_samples:
                    break
                snippet = _select_focus_sentence(chunk.text, rng)
                template = rng.choice(self._templates)
                question = template.format(snippet=snippet)
                answer = chunk.summary(320)
                strategy = rng.choice(strategies)
                metadata: Dict[str, Any] = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "strategy": strategy,
                }
                metadata.update(chunk.metadata or {})

                draft = DraftSample(
                    question=question,
                    answer=answer,
                    reference_chunk_ids=[chunk.chunk_id],
                    reference_contexts=[chunk.text],
                    strategy=strategy,
                    metadata=metadata,
                )
                drafts.append(draft)
        return drafts

    def _apply_personas_and_scenarios(
        self,
        *,
        drafts: Sequence[DraftSample],
        persona_profile: Optional[Mapping[str, Any]],
        scenarios: Sequence[ScenarioDefinition],
        rng: random.Random,
    ) -> List[DraftSample]:
        if not drafts:
            return []

        processed: List[DraftSample] = []
        for index, draft in enumerate(drafts):
            persona_result: PersonaInjectionResult = self._persona_injector.apply(
                question=draft.question,
                persona_profile=persona_profile,
                rng=rng,
            )
            candidate = draft.clone(
                question=persona_result.question,
                persona_id=persona_result.persona_id,
            )
            if scenarios:
                scenario = scenarios[index % len(scenarios)]
                question_with_scenario = _apply_scenario(candidate.question, scenario)
                candidate = candidate.clone(
                    question=question_with_scenario,
                    scenario_id=scenario.scenario_id,
                )
            processed.append(candidate)
        return processed

    @staticmethod
    def _to_testset_sample(sample: DraftSample, method: str) -> TestsetSample:
        rubrics: Dict[str, str] = {"strategy": sample.strategy}
        if sample.persona_id is not None:
            rubrics["persona_id"] = sample.persona_id
        if sample.scenario_id is not None:
            rubrics["scenario_id"] = sample.scenario_id

        eval_sample = SingleTurnSample(
            user_input=sample.question,
            reference=sample.answer,
            reference_contexts=sample.reference_contexts,
            rubrics=rubrics,
        )
        synthesizer = method or "unknown"
        return TestsetSample(eval_sample=eval_sample, synthesizer_name=synthesizer)


def _select_focus_sentence(text: str, rng: random.Random) -> str:
    candidates = _split_sentences(text)
    if not candidates:
        return text.strip()
    return rng.choice(candidates).strip()


def _split_sentences(text: str) -> List[str]:
    snippet = text.strip()
    if not snippet:
        return []
    sentences = re.split(r"(?<=[\.!?。？！])\s+", snippet)
    return [sentence for sentence in sentences if sentence]


def _apply_scenario(question: str, scenario: ScenarioDefinition) -> str:
    if scenario.instructions:
        return f"{question}\nScenario: {scenario.instructions}".strip()
    return question


__all__ = ["GenerationParameters", "GeneratorCore", "GenerationStats"]
