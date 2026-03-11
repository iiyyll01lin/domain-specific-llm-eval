from pathlib import Path

from src.interfaces.english_prompts import (
    create_custom_english_prompt,
    get_english_system_prompt,
)
from src.utils.prompt_templates import load_prompt_library, resolve_prompt_library_path


def test_prompt_library_loads_yaml_templates():
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = resolve_prompt_library_path(None, base_dir)

    library = load_prompt_library(None, base_dir)

    assert prompt_path.exists()
    assert "profiles" in library
    assert library["profiles"]["default"]["personas"]


def test_english_prompt_helpers_use_externalized_yaml():
    prompt = get_english_system_prompt("technical")

    assert "technical evaluation assistant" in prompt.lower()
    assert "english" in prompt.lower()


def test_create_custom_english_prompt_preserves_domain_requirements():
    prompt = create_custom_english_prompt(
        domain="manufacturing",
        requirements=["Use IEC-aligned terminology"],
    )

    assert "manufacturing" in prompt.lower()
    assert "IEC-aligned terminology" in prompt
