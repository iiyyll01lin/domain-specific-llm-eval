"""
KM Summary Export module (TASK-045).

Produces minimal count-based summaries for testsets and knowledge graphs
conforming to the testset_summary_v0 and kg_summary_v0 JSON schemas.
These summaries are displayed in the UI KM Summaries Panel (TASK-046).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------
TESTSET_SUMMARY_SCHEMA = "testset_summary_v0"
KG_SUMMARY_SCHEMA = "kg_summary_v0"


# ---------------------------------------------------------------------------
# Testset summary
# ---------------------------------------------------------------------------

def build_testset_summary(
    *,
    testset_id: str,
    sample_count: int,
    seed: int,
    config_hash: str,
    persona_count: int = 0,
    scenario_count: int = 0,
    duplicate: bool = False,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a testset_summary_v0 dict.

    Args:
        testset_id: Unique testset job identifier.
        sample_count: Number of Q/A samples generated.
        seed: Random seed used for generation.
        config_hash: Normalised config hash for idempotency.
        persona_count: Number of persona variants included.
        scenario_count: Number of scenario variants included.
        duplicate: Whether this was a duplicate generation (return existing).
        created_at: ISO-8601 creation timestamp.

    Returns:
        Summary dict conforming to testset_summary_v0 schema.
    """
    return {
        "schema": TESTSET_SUMMARY_SCHEMA,
        "testset_id": testset_id,
        "sample_count": sample_count,
        "seed": seed,
        "config_hash": config_hash,
        "persona_count": persona_count,
        "scenario_count": scenario_count,
        "duplicate": duplicate,
        "created_at": created_at or _now_iso(),
    }


# ---------------------------------------------------------------------------
# KG summary
# ---------------------------------------------------------------------------

def build_kg_summary(
    *,
    kg_id: str,
    node_count: int,
    relationship_count: int,
    top_entities: Optional[List[Dict[str, Any]]] = None,
    degree_histogram: Optional[List[Dict[str, Any]]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a kg_summary_v0 dict.

    Args:
        kg_id: Unique knowledge graph identifier.
        node_count: Total number of nodes in the KG.
        relationship_count: Total number of edges/relationships.
        top_entities: List of ``{"entity": str, "frequency": int}`` dicts
                      (up to 20 by convention).
        degree_histogram: Histogram bins as ``{"bin": str, "count": int}`` dicts
                          (up to 50 bins).
        created_at: ISO-8601 creation timestamp.

    Returns:
        Summary dict conforming to kg_summary_v0 schema.
    """
    histogram = (degree_histogram or [])[:50]

    return {
        "schema": KG_SUMMARY_SCHEMA,
        "kg_id": kg_id,
        "node_count": node_count,
        "relationship_count": relationship_count,
        "top_entities": (top_entities or [])[:20],
        "degree_histogram": histogram,
        "histogram_bins": len(histogram),
        "created_at": created_at or _now_iso(),
    }


# ---------------------------------------------------------------------------
# Delta computation (for UI prev-run comparison)
# ---------------------------------------------------------------------------

def compute_summary_delta(
    current: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Compute added/removed counts between current and previous summary.

    Returns a dict with ``added_*`` and ``removed_*`` fields for countable
    numeric fields.  Returns an empty dict when *previous* is ``None``.
    """
    if previous is None:
        return {}
    delta: Dict[str, Any] = {}
    count_keys = {"sample_count", "persona_count", "scenario_count",
                  "node_count", "relationship_count"}
    for key in count_keys:
        curr_val = current.get(key)
        prev_val = previous.get(key)
        if isinstance(curr_val, int) and isinstance(prev_val, int):
            diff = curr_val - prev_val
            delta[f"delta_{key}"] = diff
    return delta


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_summary(
    summary: Dict[str, Any],
    output_path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write *summary* to *output_path* as JSON."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=dest.parent, encoding=encoding
    ) as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    os.replace(tmp, dest)
    return dest


def load_summary(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load and return a previously written summary JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "TESTSET_SUMMARY_SCHEMA",
    "KG_SUMMARY_SCHEMA",
    "build_testset_summary",
    "build_kg_summary",
    "compute_summary_delta",
    "write_summary",
    "load_summary",
]
