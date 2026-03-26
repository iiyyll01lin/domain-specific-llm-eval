from __future__ import annotations

import copy

import pytest

from services.testset.config_normalizer import compute_config_hash, normalize_config


def test_normalize_removes_volatile_and_empty_fields() -> None:
    raw = {
        "method": "ragas",
        "job_id": "abc123",
        "description": "  baseline run  ",
        "options": {
            "max_samples": 200,
            "created_at": "2025-09-30T10:00:00Z",
            "persona": {
                "name": "  Analyst  ",
                "traits": ["curious", "  ", None],
                "notes": "",
            },
            "tags": [],
        },
        "extras": [None, {"value": ""}, {"value": "keep"}],
        "submitted_at": "2025-09-30T10:01:00Z",
    }

    original = copy.deepcopy(raw)
    normalized = normalize_config(raw)

    assert normalized == {
        "description": "baseline run",
        "extras": [{"value": "keep"}],
        "method": "ragas",
        "options": {
            "max_samples": 200,
            "persona": {
                "name": "Analyst",
                "traits": ["curious"],
            },
        },
    }

    # Ensure normalization does not mutate the original payload.
    assert raw == original


def test_compute_config_hash_is_order_invariant_and_strips_optionals() -> None:
    config_a = {
        "method": "ragas",
        "seed": 42,
        "persona": {
            "name": "QA",
            "languages": ["en", "zh"],
        },
        "limits": {
            "max_samples": 128,
            "min_confidence": 0.6,
        },
    }

    config_b = {
        "limits": {
            "min_confidence": 0.6,
            "max_samples": 128,
        },
        "persona": {
            "languages": ["en", "zh"],
            "name": "QA",
            "notes": None,
        },
        "seed": 42,
        "method": "ragas",
        "metadata": {},
    }

    hash_a = compute_config_hash(config_a, length=16)
    hash_b = compute_config_hash(config_b, length=16)

    assert hash_a == hash_b
    assert len(hash_a) == 16


def test_compute_config_hash_returns_full_digest_when_length_out_of_bounds() -> None:
    config = {"method": "ragas"}

    full_hash = compute_config_hash(config, length=0)
    assert len(full_hash) == 64

    full_hash_negative = compute_config_hash(config, length=-1)
    assert full_hash_negative == full_hash

    with pytest.raises(TypeError):
        # type: ignore[arg-type]
        compute_config_hash(None)
