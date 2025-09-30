from __future__ import annotations

import random

from services.testset.persona_injector import (
    PersonaInjector,
    PersonaInjectorConfig,
)


def test_persona_injection_appends_context_with_limit() -> None:
    persona_profile = {
        "id": "field-inspector",
        "name": "Field Inspector",
        "role": "Safety Assurance Specialist",
        "locale": "zh-TW",
        "description": "Focus on compliance checks, highlight remediation steps, and keep tone concise.",
    }
    config = PersonaInjectorConfig(max_token_overhead=32)
    injector = PersonaInjector(config)

    question = "Summarize the incident response procedure for manufacturing sites."
    result = injector.apply(question=question, persona_profile=persona_profile, rng=random.Random(7))

    assert result.applied is True
    assert result.persona_id == "field-inspector"
    lines = result.question.splitlines()
    assert lines[0] == question
    assert len(lines) == 2
    appended = lines[1]
    assert "Field Inspector" in appended
    assert len(appended) <= config.max_characters


def test_persona_injection_skips_when_profile_missing() -> None:
    injector = PersonaInjector()
    question = "Outline the document handling SLA."

    result = injector.apply(question=question, persona_profile=None, rng=random.Random(3))

    assert result.applied is False
    assert result.persona_id is None
    assert result.question == question
