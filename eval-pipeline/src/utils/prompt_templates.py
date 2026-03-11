"""Helpers for loading prompt, persona, and fallback templates from YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_LIBRARY: Dict[str, Any] = {
    "profiles": {
        "default": {
            "system_prompts": {
                "default": "You are a helpful assistant that must respond only in English.",
                "strict": "You must respond only in English. Never mix languages in the response.",
                "conversational": "You are a helpful assistant. Respond in clear English only.",
                "technical": "Respond in professional technical English only.",
            },
            "personas": [
                {
                    "id": "technical_specialist",
                    "name": "Technical Specialist",
                    "description": "Domain expert asking detailed technical questions.",
                    "question_style": "detailed",
                    "complexity_preference": "high",
                    "role_description": "A technical specialist who asks detailed questions about industrial processes and specifications.",
                },
                {
                    "id": "quality_inspector",
                    "name": "Quality Inspector",
                    "description": "Quality-focused reviewer asking validation questions.",
                    "question_style": "precise",
                    "complexity_preference": "medium",
                    "role_description": "A quality inspector who focuses on measurement procedures and quality control standards.",
                },
            ],
            "fallback_generation": {
                "measurement": {
                    "match_terms": ["量測", "measurement"],
                    "question_template": "What are the measurement procedures described in {title}?",
                    "answer_template": "The measurement procedures include: {excerpt}...",
                },
                "inspection": {
                    "match_terms": ["檢查", "inspection"],
                    "question_template": "What inspection steps are required according to {title}?",
                    "answer_template": "The inspection requirements are: {excerpt}...",
                },
                "general": {
                    "question_template": "What are the key points described in {title}?",
                    "answer_template": "The key information includes: {excerpt}...",
                },
            },
            "query_distribution": [
                {"synthesizer": "single_hop_specific", "weight": 0.4},
                {"synthesizer": "multi_hop_abstract", "weight": 0.4},
                {"synthesizer": "multi_hop_specific", "weight": 0.2},
            ],
        }
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_prompt_library_path(path_value: Optional[str], base_dir: Optional[Path] = None) -> Path:
    base_dir = base_dir or Path(__file__).resolve().parents[2]
    if not path_value:
        return (base_dir / "prompts" / "instruction_templates.yaml").resolve()
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


@lru_cache(maxsize=8)
def load_prompt_library(path_value: Optional[str] = None, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    library = DEFAULT_LIBRARY
    prompt_path = resolve_prompt_library_path(path_value, base_dir)
    if not prompt_path.exists():
        return library

    with open(prompt_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(library, loaded)


def get_profile(library: Dict[str, Any], profile_name: str = "default") -> Dict[str, Any]:
    profiles = library.get("profiles", {})
    return profiles.get(profile_name, profiles.get("default", {}))


def get_system_prompt(
    prompt_type: str = "default",
    *,
    profile_name: str = "default",
    library: Optional[Dict[str, Any]] = None,
) -> str:
    library = library or DEFAULT_LIBRARY
    profile = get_profile(library, profile_name)
    prompts = profile.get("system_prompts", {})
    return prompts.get(prompt_type, prompts.get("default", ""))


def get_persona_templates(
    *,
    profile_name: str = "default",
    library: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    library = library or DEFAULT_LIBRARY
    profile = get_profile(library, profile_name)
    return profile.get("personas", [])


def get_fallback_generation_templates(
    *,
    profile_name: str = "default",
    library: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    library = library or DEFAULT_LIBRARY
    profile = get_profile(library, profile_name)
    return profile.get("fallback_generation", {})


def get_query_distribution_templates(
    *,
    profile_name: str = "default",
    library: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    library = library or DEFAULT_LIBRARY
    profile = get_profile(library, profile_name)
    return profile.get("query_distribution", [])