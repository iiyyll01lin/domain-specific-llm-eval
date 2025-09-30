from __future__ import annotations

import pytest

from services.common.errors import ServiceError
from services.testset.validation import validate_testset_config


def test_validate_testset_config_returns_sanitized_payload() -> None:
    payload = {
        "method": " RAGAS ",
        "max_total_samples": 120,
        "samples_per_document": 25,
        "seed": 42,
        "selected_strategies": [" domain_specific ", "edge_cases", "DOMAIN_SPECIFIC"],
        "persona_profile": {
            "id": " analyst ",
            "name": " Ops Analyst ",
            "role": " evaluator ",
            "locale": "EN-US",
            "description": "  focus on incident reviews  ",
        },
        "extra": {"keep": True},
    }

    result = validate_testset_config(payload)

    assert result["method"] == "ragas"
    assert result["max_total_samples"] == 120
    assert result["samples_per_document"] == 25
    assert result["seed"] == 42
    # Duplicates trimmed and normalised
    assert result["selected_strategies"] == ["domain_specific", "edge_cases"]
    assert result["extra"] == {"keep": True}

    persona = result["persona_profile"]
    assert persona["id"] == "analyst"
    assert persona["name"] == "Ops Analyst"
    assert persona["role"] == "evaluator"
    assert persona["locale"] == "en-us"
    assert persona["description"] == "focus on incident reviews"


def test_validate_testset_config_rejects_unsupported_method() -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_testset_config({"method": "manual", "max_total_samples": 10})

    err = excinfo.value
    assert err.error_code == "testset_config_invalid"
    assert err.http_status == 400
    assert "unsupported method" in err.message


def test_validate_testset_config_rejects_invalid_sample_counts() -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_testset_config({"method": "ragas", "max_total_samples": 0})

    assert "max_total_samples" in excinfo.value.message

    with pytest.raises(ServiceError) as excinfo_range:
        validate_testset_config({
            "method": "ragas",
            "max_total_samples": 100,
            "samples_per_document": 101,
        })

    assert "samples_per_document" in excinfo_range.value.message


def test_validate_testset_config_rejects_seed_out_of_range() -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_testset_config({
            "method": "configurable",
            "max_total_samples": 50,
            "seed": 2**32,
        })

    assert "seed" in excinfo.value.message
