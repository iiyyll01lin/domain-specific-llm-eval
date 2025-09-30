from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class SourceChunk:
    """Lightweight representation of a processed chunk used for testset generation."""

    chunk_id: str
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self, max_characters: int = 320) -> str:
        """Return a condensed snippet of *text* trimmed to *max_characters* characters."""

        snippet = self.text.strip()
        if len(snippet) <= max_characters:
            return snippet
        return f"{snippet[: max_characters - 1].rstrip()}…"


@dataclass
class DraftSample:
    """Intermediate representation of a generated Q/A pair before final filtering."""

    question: str
    answer: str
    reference_chunk_ids: List[str]
    reference_contexts: List[str]
    strategy: str
    persona_id: Optional[str] = None
    scenario_id: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "question": self.question,
            "answer": self.answer,
            "reference_chunk_ids": list(self.reference_chunk_ids),
            "reference_contexts": list(self.reference_contexts),
            "strategy": self.strategy,
        }
        if self.persona_id is not None:
            payload["persona_id"] = self.persona_id
        if self.scenario_id is not None:
            payload["scenario_id"] = self.scenario_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def clone(
        self,
        *,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        reference_chunk_ids: Optional[Iterable[str]] = None,
        reference_contexts: Optional[Iterable[str]] = None,
        strategy: Optional[str] = None,
        persona_id: Optional[str] = ...,
        scenario_id: Optional[str] = ...,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DraftSample":
        """Return a shallow copy replacing provided fields."""

        next_instance = replace(
            self,
            question=self.question if question is None else question,
            answer=self.answer if answer is None else answer,
            reference_chunk_ids=list(reference_chunk_ids)
            if reference_chunk_ids is not None
            else list(self.reference_chunk_ids),
            reference_contexts=list(reference_contexts)
            if reference_contexts is not None
            else list(self.reference_contexts),
            strategy=self.strategy if strategy is None else strategy,
            persona_id=self.persona_id if persona_id is ... else persona_id,
            scenario_id=self.scenario_id if scenario_id is ... else scenario_id,
        )
        if metadata is not None:
            next_instance.metadata = dict(metadata)
        else:
            next_instance.metadata = dict(self.metadata)
        return next_instance


__all__ = ["DraftSample", "SourceChunk"]
