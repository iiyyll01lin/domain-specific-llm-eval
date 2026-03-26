from __future__ import annotations

from services.testset.persona import build_personas_document, build_scenarios_document


def test_build_personas_document_generates_slug_when_missing_id() -> None:
    persona_payload = {"name": "Quality Auditor", "role": "Auditor", "locale": "en-US"}

    document = build_personas_document(
        persona_payload,
        job_id="job-123",
        config_hash="hash-xyz",
        seed=42,
        generated_at="2025-09-30T12:00:00Z",
    )

    assert document["job_id"] == "job-123"
    assert document["count"] == 1
    assert document["items"][0]["id"] == "quality-auditor"
    assert document["items"][0]["locale"] == "en-US"


def test_build_personas_document_handles_empty_profile() -> None:
    document = build_personas_document(
        None,
        job_id="job-123",
        config_hash="hash-xyz",
        seed=None,
        generated_at="2025-09-30T12:00:00Z",
    )

    assert document["count"] == 0
    assert document["items"] == []


def test_build_scenarios_document_normalises_entries() -> None:
    scenarios = [
        {"scenario_id": "s-1", "label": "Primary", "instructions": "Follow the SOP", "priority": "high"},
        {"label": "Fallback", "instructions": "Fallback instructions."},
    ]

    document = build_scenarios_document(
        scenarios,
        job_id="job-123",
        config_hash="hash-xyz",
        seed=101,
        generated_at="2025-09-30T12:00:00Z",
    )

    assert document["count"] == 2
    assert document["items"][0]["scenario_id"] == "s-1"
    assert document["items"][0]["extras"] == {"priority": "high"}
    assert document["items"][1]["scenario_id"] == "scenario-2"
    assert document["items"][1]["label"] == "Fallback"
