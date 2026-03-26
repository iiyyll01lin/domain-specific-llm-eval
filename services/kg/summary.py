"""KG summary computation.

TASK-063: counts, degree histogram (≤50 bins), top entities with frequency.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


MAX_HISTOGRAM_BINS = 50
DEFAULT_TOP_ENTITIES = 20


def _compute_degrees(
    nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
) -> Dict[str, int]:
    degree: Dict[str, int] = {n["node_id"]: 0 for n in nodes}
    for rel in relationships:
        for key in ("source", "target"):
            nid = rel.get(key)
            if nid in degree:
                degree[nid] += 1
    return degree


def compute_degree_histogram(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    max_bins: int = MAX_HISTOGRAM_BINS,
) -> List[Dict[str, Any]]:
    """Return degree histogram with at most *max_bins* bins."""
    if not nodes:
        return []
    degree = _compute_degrees(nodes, relationships)
    values = list(degree.values())
    min_d, max_d = min(values), max(values)
    if min_d == max_d:
        return [{"bin_start": min_d, "bin_end": max_d, "count": len(values)}]
    n_bins = min(max_bins, max_d - min_d + 1)
    bin_size = (max_d - min_d) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - min_d) / bin_size), n_bins - 1)
        counts[idx] += 1
    histogram = []
    for i, c in enumerate(counts):
        histogram.append(
            {
                "bin_start": round(min_d + i * bin_size, 2),
                "bin_end": round(min_d + (i + 1) * bin_size, 2),
                "count": c,
            }
        )
    return histogram


def get_top_entities(
    nodes: List[Dict[str, Any]], limit: int = DEFAULT_TOP_ENTITIES
) -> List[Dict[str, Any]]:
    """Aggregate entity frequency across all nodes, return top *limit*."""
    freq: Dict[str, int] = defaultdict(int)
    for node in nodes:
        for ent in node.get("entities", []):
            freq[ent] += 1
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])[:limit]
    return [{"entity": ent, "count": cnt} for ent, cnt in ranked]


def build_kg_summary(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    kg_id: str,
    top_entity_limit: int = DEFAULT_TOP_ENTITIES,
) -> Dict[str, Any]:
    """Build the full JSON summary payload for a KG job."""
    histogram = compute_degree_histogram(nodes, relationships)
    top_entities = get_top_entities(nodes, limit=top_entity_limit)
    return {
        "kg_id": kg_id,
        "node_count": len(nodes),
        "edge_count": len(relationships),
        "top_entities": top_entities,
        "degree_histogram": histogram,
        "histogram_bins": len(histogram),
    }
