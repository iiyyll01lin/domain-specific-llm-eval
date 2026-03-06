"""Relationship builders for the KG service.

TASK-062b/c: Jaccard (entity overlap), Overlap (keyphrase containment),
Cosine (embedding similarity, optional), SummaryCosine (optional).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _overlap(a: List[str], b: List[str]) -> float:
    """Overlap score: |A ∩ B| / min(|A|, |B|) — asymmetric containment."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / min(len(sa), len(sb))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------


def build_jaccard_relationships(
    nodes: List[Dict[str, Any]],
    threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Build entity-based Jaccard similarity relationships."""
    relationships: List[Dict[str, Any]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ea = nodes[i].get("entities", [])
            eb = nodes[j].get("entities", [])
            score = _jaccard(ea, eb)
            if score >= threshold:
                relationships.append(
                    {
                        "source": nodes[i]["node_id"],
                        "target": nodes[j]["node_id"],
                        "type": "jaccard_similarity",
                        "score": round(score, 4),
                    }
                )
    return relationships


def build_overlap_relationships(
    nodes: List[Dict[str, Any]],
    threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Build keyphrase-based overlap relationships."""
    relationships: List[Dict[str, Any]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ka = nodes[i].get("keyphrases", [])
            kb = nodes[j].get("keyphrases", [])
            score = _overlap(ka, kb)
            if score >= threshold:
                relationships.append(
                    {
                        "source": nodes[i]["node_id"],
                        "target": nodes[j]["node_id"],
                        "type": "overlap_score",
                        "score": round(score, 4),
                    }
                )
    return relationships


def build_cosine_relationships(
    nodes: List[Dict[str, Any]],
    threshold: float = 0.7,
    embedding_key: str = "embedding",
) -> Tuple[List[Dict[str, Any]], int]:
    """Build cosine similarity relationships from embeddings.

    Returns (relationships, skipped_count) — nodes without embeddings are
    skipped gracefully (TASK-062c).
    """
    relationships: List[Dict[str, Any]] = []
    skipped = 0
    for i in range(len(nodes)):
        ea = nodes[i].get(embedding_key)
        if not ea:
            skipped += 1
            continue
        for j in range(i + 1, len(nodes)):
            eb = nodes[j].get(embedding_key)
            if not eb:
                continue
            score = _cosine(ea, eb)
            if score >= threshold:
                relationships.append(
                    {
                        "source": nodes[i]["node_id"],
                        "target": nodes[j]["node_id"],
                        "type": f"cosine_{embedding_key}",
                        "score": round(score, 4),
                    }
                )
    return relationships, skipped


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def build_all_relationships(
    nodes: List[Dict[str, Any]],
    jaccard_threshold: float = 0.1,
    overlap_threshold: float = 0.05,
    cosine_threshold: float = 0.7,
    summary_cosine_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run all relationship builders and return the merged list."""
    results: List[Dict[str, Any]] = []
    results.extend(build_jaccard_relationships(nodes, threshold=jaccard_threshold))
    results.extend(build_overlap_relationships(nodes, threshold=overlap_threshold))
    cosine_rels, _skipped = build_cosine_relationships(
        nodes, threshold=cosine_threshold, embedding_key="embedding"
    )
    results.extend(cosine_rels)
    summary_rels, _skipped2 = build_cosine_relationships(
        nodes, threshold=summary_cosine_threshold, embedding_key="summary_embedding"
    )
    results.extend(summary_rels)
    return results


def count_by_type(relationships: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rel in relationships:
        t = rel.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts
