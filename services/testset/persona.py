from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def build_personas_document(
    persona: Optional[Mapping[str, Any]],
    *,
    job_id: str,
    config_hash: str,
    seed: Optional[int],
    generated_at: str,
) -> Dict[str, Any]:
    """Return a normalized personas manifest for downstream consumption."""

    items: list[Dict[str, Any]] = []
    if persona:
        items.append(_normalise_persona(persona))

    return {
        "job_id": job_id,
        "config_hash": config_hash,
        "seed": seed,
        "generated_at": generated_at,
        "count": len(items),
        "items": items,
    }


def build_scenarios_document(
    scenarios: Sequence[Mapping[str, Any]] | Sequence[Any] | None,
    *,
    job_id: str,
    config_hash: str,
    seed: Optional[int],
    generated_at: str,
) -> Dict[str, Any]:
    """Return a normalized scenarios manifest for downstream consumption."""

    items: list[Dict[str, Any]] = []
    if scenarios:
        for index, scenario in enumerate(scenarios, start=1):
            if isinstance(scenario, Mapping):
                items.append(_normalise_scenario(scenario, index=index))
            else:
                items.append(
                    {
                        "scenario_id": f"scenario-{index}",
                        "label": str(scenario),
                        "instructions": str(scenario),
                    }
                )

    return {
        "job_id": job_id,
        "config_hash": config_hash,
        "seed": seed,
        "generated_at": generated_at,
        "count": len(items),
        "items": items,
    }


def _normalise_persona(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = {str(key): value for key, value in payload.items()}
    persona_id = str(data.get("id") or "").strip()
    if not persona_id:
        fallback_source = str(data.get("name") or data.get("role") or "persona").strip()
        persona_id = _slugify(fallback_source or "persona")
    data["id"] = persona_id
    return data


def _normalise_scenario(payload: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    scenario_id = str(payload.get("scenario_id") or "").strip()
    if not scenario_id:
        scenario_id = f"scenario-{index}"
    label = str(payload.get("label") or scenario_id).strip()
    instructions = str(payload.get("instructions") or "").strip()
    extras = {
        key: value
        for key, value in payload.items()
        if key not in {"scenario_id", "label", "instructions"}
    }
    document: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "label": label,
        "instructions": instructions,
    }
    if extras:
        document["extras"] = extras
    return document


def _slugify(value: str) -> str:
    candidate = _SLUG_RE.sub("-", value.lower()).strip("-")
    return candidate or "persona"


__all__ = [
    "build_personas_document",
    "build_scenarios_document",
]
