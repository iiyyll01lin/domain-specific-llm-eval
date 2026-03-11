from pathlib import Path

from src.utils.prompt_templates import get_system_prompt, load_prompt_library


def get_english_system_prompt(prompt_type: str = "default") -> str:
    """
    Get predefined English system prompt by type.

    Args:
        prompt_type: Type of prompt ('default', 'strict', 'conversational', 'technical')

    Returns:
        English system prompt string
    """
    library = load_prompt_library(base_dir=Path(__file__).resolve().parents[2])
    return get_system_prompt(prompt_type, library=library)


def create_custom_english_prompt(domain: str = "", requirements: list = None) -> str:
    """
    Create a custom English system prompt for specific domains.

    Args:
        domain: Specific domain (e.g., 'medical', 'legal', 'technical')
        requirements: Additional requirements list

    Returns:
        Custom English system prompt
    """
    library = load_prompt_library(base_dir=Path(__file__).resolve().parents[2])
    base = (
        get_system_prompt("technical", library=library)
        or "You are a helpful assistant that must respond only in English."
    )

    if domain:
        base += f"\nYou specialize in {domain} topics."

    base += """\n
LANGUAGE REQUIREMENTS:
- Always respond in English language only
- Translate non-English inputs internally before answering
- Use appropriate terminology for the domain
- Maintain professional tone"""

    if requirements:
        base += "\n\nADDITIONAL REQUIREMENTS:\n"
        for req in requirements:
            base += f"- {req}\n"

    base += "\nRemember: ALL responses must be in English only."

    return base
