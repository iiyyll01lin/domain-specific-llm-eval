# ADR-003: Subgraph Sampling Strategy for Knowledge Graph API

Status: Accepted  
Date: 2025-09-10  
Decision Owners: Data Engineering Lead, Platform Lead  
Reviewers: Frontend Lead, Performance SME  

## 1. Context
The Subgraph API (spec §27) must return deterministic, size-bounded subgraphs to support focused visualization without overloading client or server. Requirements: reproducibility (same focus + parameters → same subgraph), bounded node/edge counts, semantic relevance preservation, graceful indication of truncation, and anti-abuse (cost control).

## 2. Decision
Adopt a layered sampling pipeline:
1. Anchor Selection: Start from focus entity/entities (or top-degree hubs for global view) → anchor set A.
2. Neighborhood Expansion: Include 1-hop neighbors; if budget remains, apply relevance-sorted 2nd hop.
3. Scoring & Ordering: Score candidate nodes by composite metric (degree_norm * w1 + rel_score_avg * w2 + entity_salience * w3).
4. Deterministic Hash Ordering: Generate stable hash(seed, node_id) for tie-breaking & reproducibility.
5. Budget Truncation: Enforce max_nodes (default 500), max_edges scaled (~3 * max_nodes).
6. Truncation Flag: Set truncated=true if candidates exceed budget.

## 3. Rationale
- Balances semantic relevance (relationship scores) and structural importance (degree, salience).
- Deterministic tie-breaking ensures UI caching and stable cognitive map.
- Explicit budgets simplify performance bounds & security review.

## 4. Alternatives Considered
| Approach                       | Pros                   | Cons                                   | Verdict            |
|--------------------------------|------------------------|----------------------------------------|--------------------|
| Pure BFS                       | Simple                 | May include low-relevance tail quickly | Rejected (quality) |
| Random Walk                    | Diverse context        | Non-deterministic w/out extra control  | Rejected (repro)   |
| Community Detection Precompute | High semantic grouping | Precompute overhead, complexity        | Deferred           |

## 5. API Parameters (Initial)
`GET /kg/{kg_id}/subgraph?focus=...&max_nodes=500&seed=123&hops=2`
- focus: comma-separated node IDs (optional)
- max_nodes: int (≤500 hard cap server-enforced)
- seed: int (default stable global seed)
- hops: expansion depth (1 or 2)

## 6. Output Schema (Draft)
```
{
  "nodes": [ {"id", "label", "degree", "entity_type", "salience"} ],
  "edges": [ {"id", "source", "target", "rel_type", "score"} ],
  "stats": {"truncated": bool, "focus_count": int, "expansion_hops": int}
}
```

## 7. Determinism Anchors
- Node ordering: stable sort by (score DESC, hash(seed,node_id) ASC)
- Edge filtering: retain only edges whose both endpoints selected.
- Hash function: SHA256(seed||node_id) truncated to 64 bits.

## 8. Risks & Mitigations
| Risk                                | Impact               | Mitigation                            |
|-------------------------------------|----------------------|---------------------------------------|
| Focus on hub yields dense explosion | Resource spike       | Budget truncation + hop limit         |
| Seed abuse (probing)                | Cache fragmentation  | Restrict seed range or rate limit     |
| Score weight mis-tuning             | Low-quality subgraph | Configurable weights + metrics review |

## 9. Metrics / Success Criteria
- P95 latency < 300ms for max_nodes=500.
- Repeated identical requests produce byte-identical node ID sequence.
- Client memory usage stable (<50MB) for 500-node render.

## 10. Future Extensions
- Community-aware sampling mode (phase 2).
- Edge type filtering parameter.
- Server-side caching keyed by (kg_id, focus_set_hash, seed, max_nodes, hops, weight_profile).

---
Status will change to Accepted after prototype validation & performance benchmark (TASK-066).
