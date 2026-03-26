from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

# Keys that should be excluded from the normalized configuration because they
# represent volatile runtime metadata that must not influence determinism.
_VOLATILE_KEYS = {
    "job_id",
    "request_id",
    "trace_id",
    "created_at",
    "updated_at",
    "submitted_at",
    "completed_at",
    "timestamp",
    "ts",
}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return all(_is_empty(item) for item in value)
    if isinstance(value, dict):
        return all(_is_empty(item) for item in value.values())
    return False


def _sortable_key(item: Any) -> str:
    if isinstance(item, (dict, list)):
        try:
            return json.dumps(item, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return repr(item)
    return repr(item)


def _normalize_sequence(sequence: Iterable[Any]) -> list[Any]:
    normalized_items = []
    for item in sequence:
        normalized = _normalize_value(item)
        if not _is_empty(normalized):
            normalized_items.append(normalized)
    return normalized_items


def _normalize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _VOLATILE_KEYS:
            continue
        normalized_value = _normalize_value(value)
        if _is_empty(normalized_value):
            continue
        normalized[key] = normalized_value
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, set):
        normalized_items = _normalize_sequence(value)
        return sorted(normalized_items, key=_sortable_key)
    if isinstance(value, (list, tuple)):
        return _normalize_sequence(value)
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized representation of *config* suitable for hashing.

    Normalization applies the following rules recursively:

    - Volatile runtime keys (timestamps, request identifiers, etc.) are removed.
    - Empty optionals (``None``, empty containers, blank strings) are dropped.
    - Strings are trimmed.
    - Dictionaries are re-created with keys sorted lexicographically so that
      ``json.dumps`` produces a stable representation regardless of input order.
    """

    if not isinstance(config, Mapping):  # type: ignore[arg-type]
        raise TypeError("config must be a mapping")

    normalized = _normalize_mapping(config)
    # Ensure nested dictionaries are sorted by key for deterministic dumps.
    return _sort_nested(normalized)


def _sort_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_nested(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_nested(item) for item in value]
    return value


def compute_config_hash(config: Mapping[str, Any], *, length: int = 12) -> str:
    """Compute a deterministic hash for *config*.

    The hash is derived from the SHA-256 digest of the normalized JSON
    representation and truncated to ``length`` characters (default: 12).
    """

    normalized = normalize_config(config)
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if length <= 0 or length > len(digest):
        return digest
    return digest[:length]


__all__ = [
    "normalize_config",
    "compute_config_hash",
]
