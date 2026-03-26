from __future__ import annotations

import pytest

from src.evaluation.deterministic_backends import (
    DeterministicIntentBackend,
    DeterministicSpatialBackend,
    DeterministicSymbolicBackend,
    DeterministicTemporalBackend,
)


@pytest.mark.parametrize(
    ("answer", "context", "was_proven", "expected_fact", "expected_score"),
    [
        ("Paris is the capital of France.", "Paris is the capital of France.", True, "Paris", 1.0),
        ("France has a major city.", "Paris is the capital of France.", True, "Paris", 0.05),
        ("Paris is mentioned in the context.", "Paris is the capital of France.", False, "", 0.3333),
    ],
)
def test_symbolic_backend_contract(answer, context, was_proven, expected_fact, expected_score) -> None:
    backend = DeterministicSymbolicBackend()
    result = backend.evaluate(answer, context, was_proven, expected_fact)

    assert round(result["score"], 4) == round(expected_score, 4)
    assert result["contract"]["backend"] == "deterministic_symbolic"


@pytest.mark.parametrize(
    ("query", "coordinates", "answer", "context", "expected_score"),
    [
        ("What is here?", (10, 5, 2), "A robotic assembly arm is nearby.", "A robotic assembly arm.", 1.0),
        ("What is here?", (10, 5, 2), "This zone has an assembly asset.", "A robotic assembly arm.", 0.7),
        ("What is here?", (10, 5, 2), "There is a beach here.", "A robotic assembly arm.", 0.0),
    ],
)
def test_spatial_backend_contract(query, coordinates, answer, context, expected_score) -> None:
    backend = DeterministicSpatialBackend()
    result = backend.evaluate(query, coordinates, answer, context)

    assert result["score"] == expected_score
    assert result["contract"]["coordinates"] == coordinates


@pytest.mark.parametrize(
    ("eeg_vector", "expected_intent"),
    [([2.0, 4.0, 1.0], "URGENT_REQUEST"), ([1.0, 1.0, 1.0], "GENERAL_INQUIRY")],
)
def test_intent_backend_decode_contract(eeg_vector, expected_intent) -> None:
    backend = DeterministicIntentBackend()
    result = backend.decode(eeg_vector)

    assert result["intent"] == expected_intent
    assert result["contract"]["backend"] == "deterministic_intent"


@pytest.mark.parametrize(
    ("intent", "response", "expected_score"),
    [
        ("URGENT_REQUEST", "We must act immediately.", 1.0),
        ("GENERAL_INQUIRY", "Here is a calm response.", 0.9),
        ("GENERAL_INQUIRY", "Act immediately now.", 0.2),
    ],
)
def test_intent_backend_alignment_contract(intent, response, expected_score) -> None:
    backend = DeterministicIntentBackend()
    result = backend.evaluate(intent, response)

    assert result["score"] == expected_score
    assert result["contract"]["intent"] == intent


@pytest.mark.parametrize(
    ("timeline", "prediction", "expected_score"),
    [
        (["market crash"], "Inflation drops after the market crash.", 0.95),
        (["supply chain disruption"], "The disruption affects supply chain capacity.", 0.6),
        (["market crash"], "Birds are blue.", 0.1),
    ],
)
def test_temporal_backend_contract(timeline, prediction, expected_score) -> None:
    backend = DeterministicTemporalBackend()
    result = backend.evaluate(timeline, prediction)

    assert result["score"] == expected_score
    assert result["contract"]["backend"] == "deterministic_temporal"