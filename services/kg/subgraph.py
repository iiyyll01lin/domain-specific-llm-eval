"""Deterministic subgraph sampling.

TASK-066: read-only, reproducible BFS from a seed entity up to max_nodes.
Returns a truncated flag when the full neighbourhood exceeds the cap.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional


MAX_NODES_CAP = 500


def sample_subgraph(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    seed_entity: Optional[str] = None,
    max_nodes: int = MAX_NODES_CAP,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return a deterministic subgraph rooted at *seed_entity*.

    If *seed_entity* is None or not found, a random node is chosen using
    *seed* for reproducibility.  The BFS traversal order is randomised with
    *seed* so repeated calls with the same inputs always return the same
    subset.

    Return shape::

        {
            "nodes": [...],
            "relationships": [...],
            "truncated": bool,
            "seed_node_id": str,
        }
    """
    if not nodes:
        return {"nodes": [], "relationships": [], "truncated": False, "seed_node_id": None}

    # Build adjacency index
    node_by_id: Dict[str, Dict[str, Any]] = {n["node_id"]: n for n in nodes}
    adj: Dict[str, List[str]] = {nid: [] for nid in node_by_id}
    for rel in relationships:
        s, t = rel.get("source"), rel.get("target")
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)

    # Resolve seed node
    seed_node_id: Optional[str] = None
    if seed_entity:
        for node in nodes:
            if seed_entity in node.get("entities", []):
                seed_node_id = node["node_id"]
                break
        if seed_node_id is None and seed_entity in node_by_id:
            seed_node_id = seed_entity

    if seed_node_id is None:
        rng = random.Random(seed)
        seed_node_id = rng.choice(list(node_by_id.keys()))

    # BFS with deterministic ordering
    visited: list[str] = []
    queue: deque[str] = deque([seed_node_id])
    seen = {seed_node_id}
    rng = random.Random(seed)

    while queue and len(visited) < max_nodes:
        current = queue.popleft()
        visited.append(current)
        neighbours = sorted(adj[current])  # deterministic sorted base
        rng.shuffle(neighbours)  # then shuffle with fixed seed
        for nbr in neighbours:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)

    truncated = len(visited) < len(nodes)
    selected_set = set(visited)
    sub_nodes = [node_by_id[nid] for nid in visited]
    sub_rels = [
        rel
        for rel in relationships
        if rel.get("source") in selected_set and rel.get("target") in selected_set
    ]

    return {
        "nodes": sub_nodes,
        "relationships": sub_rels,
        "truncated": truncated,
        "seed_node_id": seed_node_id,
    }
