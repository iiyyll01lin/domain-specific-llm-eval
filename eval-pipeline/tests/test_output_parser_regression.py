from __future__ import annotations

from pydantic import BaseModel

from src.utils.output_parser_fix import (
    apply_ragas_output_parser_fixes,
    sanitize_json_output,
)


def test_apply_ragas_output_parser_fixes_returns_success() -> None:
    assert apply_ragas_output_parser_fixes() is True


def test_sanitize_json_output_handles_empty_and_partial_json() -> None:
    assert sanitize_json_output("") == '{"text": ""}'
    assert sanitize_json_output("{}") == '{"text": ""}'
    sanitized = sanitize_json_output('{"valid": "json"}')
    assert '"text"' in sanitized


def test_pydantic_model_creation_still_works_after_patch() -> None:
    apply_ragas_output_parser_fixes()

    class TestModel(BaseModel):
        name: str = "default"
        value: int = 0

    model = TestModel(name="demo", value=1)
    assert model.name == "demo"
    assert model.value == 1