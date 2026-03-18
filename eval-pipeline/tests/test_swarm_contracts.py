from __future__ import annotations

from src.evaluation.swarm_agent import SwarmSynthesizer


def test_swarm_contract_policy_answer_requires_revision() -> None:
    synthesizer = SwarmSynthesizer()

    verdict = synthesizer.debate_answer(
        "What is the security policy?",
        "The system should probably rotate credentials.",
    )

    assert verdict["agreement_rate"] < 1.0
    assert verdict["contract"]["backend"] == "deterministic_swarm"
    assert verdict["contract"]["revision_required"] is True


def test_swarm_contract_sensitive_content_triggers_security_revision() -> None:
    synthesizer = SwarmSynthesizer()

    verdict = synthesizer.debate_answer(
        "Summarize the incident",
        "The password and pii were stored in the answer.",
    )

    assert verdict["contract"]["sensitive_content_detected"] is True
    assert verdict["dissent_reasons"]


def test_swarm_contract_approves_safe_answer() -> None:
    synthesizer = SwarmSynthesizer()

    verdict = synthesizer.debate_answer(
        "Summarize the implementation",
        "The implementation documents the workflow and cites the evidence clearly.",
    )

    assert verdict["agreement_rate"] == 1.0
    assert verdict["final_answer"].endswith("[Debated and Approved]")