from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


DEFAULT_MAX_TOKEN_OVERHEAD = 64
_ASCII_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass(frozen=True)
class PersonaInjectionResult:
    question: str
    persona_id: Optional[str]
    applied: bool


@dataclass(frozen=True)
class PersonaInjectorConfig:
    max_token_overhead: int = DEFAULT_MAX_TOKEN_OVERHEAD
    prefix: str = "Persona context:"

    @property
    def max_characters(self) -> int:
        # Approximate 1 token ≈ 4 chars (conservative upper bound)
        return max(24, self.max_token_overhead * 4)


class PersonaInjector:
    """Appends concise persona context to the generated question prompts."""

    def __init__(self, config: PersonaInjectorConfig | None = None) -> None:
        self._config = config or PersonaInjectorConfig()

    def apply(
        self,
        *,
        question: str,
        persona_profile: Optional[Mapping[str, Any]],
        rng: random.Random,
    ) -> PersonaInjectionResult:
        if not persona_profile:
            return PersonaInjectionResult(question=question, persona_id=None, applied=False)

        persona_id = str(persona_profile.get("id") or _slugify(str(persona_profile.get("name", "persona"))))
        description = self._build_description(persona_profile)
        if not description:
            return PersonaInjectionResult(question=question, persona_id=persona_id, applied=False)

        appended = self._format_context(description, persona_id, rng)
        if not appended:
            return PersonaInjectionResult(question=question, persona_id=persona_id, applied=False)

        enriched_question = "\n".join([line for line in (question.strip(), appended) if line])
        return PersonaInjectionResult(question=enriched_question, persona_id=persona_id, applied=True)

    def _build_description(self, persona_profile: Mapping[str, Any]) -> str:
        segments: list[str] = []
        name = _clean_value(persona_profile.get("name"))
        role = _clean_value(persona_profile.get("role"))
        locale = _clean_value(persona_profile.get("locale"))
        description = _clean_value(persona_profile.get("description"))

        if name:
            segments.append(f"Name {name}")
        if role:
            segments.append(f"Role {role}")
        if locale:
            segments.append(f"Locale {locale}")
        if description:
            segments.append(description)
        return ". ".join(segments).strip()

    def _format_context(self, description: str, persona_id: str, rng: random.Random) -> str:
        message = f"{self._config.prefix} [{persona_id}]: {description}"
        if len(message) <= self._config.max_characters:
            return message

        # Deterministically trim description length while preserving key metadata
        max_desc_length = max(0, self._config.max_characters - len(self._config.prefix) - len(persona_id) - 6)
        truncated = _truncate(description, max_desc_length)
        if not truncated:
            return ""
        variants = [
            f"{self._config.prefix} [{persona_id}]: {truncated}",
            f"{self._config.prefix} ({persona_id}) {truncated}",
        ]
        return rng.choice(variants)


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _slugify(value: str) -> str:
    candidate = _ASCII_RE.sub("-", value.lower()).strip("-")
    return candidate or "persona"


__all__ = [
    "PersonaInjectionResult",
    "PersonaInjector",
    "PersonaInjectorConfig",
]
