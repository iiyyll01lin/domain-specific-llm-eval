# Technical Specification — Graph Context Relevance & Evaluation Algorithms

> **Project:** Domain-Specific RAG Evaluation & MLOps Platform  
> **Version:** 1.1.0  
> **Branch:** `feat/graph-context-relevance`  
> **Last updated:** 2026-03-27

---

## Table of Contents

1. [Overview](#overview)
2. [Graph Context Relevance (GCR) — Composite Score](#graph-context-relevance-gcr--composite-score)
3. [Component: Entity Overlap ($S_e$)](#component-entity-overlap-s_e)
4. [Component: Structural Connectivity ($S_c$)](#component-structural-connectivity-s_c)
5. [Component: Hub Noise Penalty ($P_h$)](#component-hub-noise-penalty-p_h)
6. [Complexity Analysis](#complexity-analysis)
7. [Weight Calibration](#weight-calibration)
8. [Edge-Case Behaviour](#edge-case-behaviour)
9. [Two-Tier Entity Highlighter](#two-tier-entity-highlighter)
10. [Output Contract](#output-contract)
11. [Related Files](#related-files)

---

## Overview

The **Graph Context Relevance (GCR)** metric evaluates the *topological quality* of a retrieved
subgraph relative to a question/answer pair. Unlike cosine-similarity-based metrics, which treat
retrieved context as an unordered bag of embeddings, GCR measures whether the retrieval forms a
**coherent, connected neighbourhood** in the knowledge graph.

GCR is:
- **100% offline** — no LLM calls, no embedding model calls at evaluation time.
- **Deterministic** — identical inputs always produce identical outputs.
- **O(N + E)** — scales linearly with graph size on the first call; cached thereafter.

The evaluator is implemented in `eval-pipeline/src/evaluation/graph_context_relevance.py` and
reads from any backend that satisfies the `GraphStore` Protocol
(`eval-pipeline/src/utils/graph_store.py`).

---

## Graph Context Relevance (GCR) — Composite Score

$$
\boxed{\mathrm{GCR} = \mathrm{clip}\!\left(\alpha \cdot S_e + \beta \cdot S_c - \gamma \cdot P_h,\ 0.0,\ 1.0\right)}
$$

| Symbol | Name | Range | Default weight |
|--------|------|--------|----------------|
| $S_e$ | Entity Overlap | $[0, 1]$ | $\alpha = 0.4$ |
| $S_c$ | Structural Connectivity | $[0, 1]$ | $\beta = 0.4$ |
| $P_h$ | Hub Noise Penalty | $[0, 1]$ | $\gamma = 0.2$ |

The `clip` operation keeps the composite score in $[0.0, 1.0]$ regardless of weight configuration.

**Maximum achievable score** (when $S_e = S_c = 1$, $P_h = 0$):

$$
\mathrm{GCR}_{\max} = \alpha + \beta = 0.4 + 0.4 = 0.8
$$

The ceiling is intentionally below 1.0 to leave headroom for domain-calibrated weight
configurations where $\alpha + \beta > 1$.

---

## Component: Entity Overlap ($S_e$)

**Definition:** Mean Jaccard similarity between the joint question+answer token set and each
retrieved node's token set.

$$
S_e = \frac{1}{|R|} \sum_{n \in R} \mathrm{Jaccard}(Q, T_n)
$$

where:
- $Q = \text{tokens}(\text{question} \cup \text{expected\_answer})$ — lowercase alphanumeric tokens
- $T_n = \text{tokens}(\text{keyphrases}_n \cup \text{entities}_n \cup \text{content}_n)$
- $\mathrm{Jaccard}(A, B) = \dfrac{|A \cap B|}{|A \cup B|}$, with $\mathrm{Jaccard}(\emptyset, \cdot) = 0$

**Tokenisation rule:** `re.findall(r"[a-z0-9_\u4e00-\u9fff]+", text.lower())` — captures
ASCII alphanumeric, underscore, and CJK unified ideographs (U+4E00–U+9FFF), supporting
Chinese/English mixed-language corpora.

**Complexity:** $O(|R| \cdot |Q|)$ — linear in the number of retrieved nodes times query token set size.

**Semantics:** $S_e$ measures *lexical relevance*. A retrieval with all semantically relevant but
lexically identical nodes scores $S_e = 1$. A retrieval of unrelated nodes scores near 0.

---

## Component: Structural Connectivity ($S_c$)

**Definition:** Fraction of retrieved nodes in the largest connected component of the induced
subgraph.

$$
S_c = \frac{\left|\mathrm{LCC}(G[R])\right|}{|R|}
$$

where:
- $G[R]$ is the undirected subgraph induced by the retrieved node set $R$
- $\mathrm{LCC}(\cdot)$ is the largest connected component (by node count)
- Only edges whose *both* endpoints are in $R$ are included (orphaned edges are excluded)

**Special cases:**

| Condition | $S_c$ value | Rationale |
|-----------|-------------|-----------|
| $\|R\| = 0$ | $0.0$ | Empty retrieval |
| $\|R\| = 1$ | $1.0$ | Single node is trivially connected |
| $\|R\| > 1$, all disconnected | $1 / \|R\|$ | Worst case: each node its own component |
| All nodes in one component | $1.0$ | Perfect structural coherence |

**Complexity:** $O(|R| + |E_R|)$ — BFS/DFS over the induced subgraph via `networkx.connected_components`.

**Semantics:** $S_c$ rewards retrievers that pull *coherent neighbourhoods*. A retriever fetching
5 unrelated document chunks scores $S_c = 0.2$, while one fetching a path through the graph scores
$S_c = 1.0$.

---

## Component: Hub Noise Penalty ($P_h$)

**Definition:** Fraction of retrieved nodes classified as *degree hubs* in the full graph.

$$
P_h = \frac{|\{n \in R : \deg_G(n) > \mu_{\deg} + 2\sigma_{\deg}\}|}{|R|}
$$

where:
- $\mu_{\deg}$ and $\sigma_{\deg}$ are the mean and sample standard deviation of all node degrees in the full graph $G$
- The $\mu + 2\sigma$ threshold corresponds to the 97.7th percentile under a normal degree distribution

**Guard conditions:**

| Condition | $P_h$ value | Rationale |
|-----------|-------------|-----------|
| $\|R\| = 0$ | $0.0$ | Empty retrieval |
| $\|G\| < 2$ | $0.0$ | Cannot compute meaningful statistics |
| $\sigma_{\deg} = 0$ | $0.0$ | Regular graph — no hub structure |
| $\sigma_{\deg} > 0$, hubs found | $> 0$ | Penalty proportional to hub fraction |

**Complexity:** $O(N)$ for degree statistics, $O(|R|)$ for hub classification.

**Semantics:** $P_h$ penalises retrieval of high-connectivity "concept hub" nodes (e.g., the node
for "steel plate" in a manufacturing QA corpus) that appear as relevant to nearly every query but
add no discriminative context. These are identified statistically, not by domain heuristics.

---

## Complexity Analysis

| Phase | Operation | Complexity | Notes |
|-------|-----------|-----------|-------|
| Graph build (first call) | Load N nodes + E edges from SQLite | $O(N + E)$ | Cached; subsequent calls are $O(1)$ |
| Graph build | `nx.Graph.add_node / add_edge` | $O(N + E)$ amortised | Dict-of-dicts; each op is $O(1)$ amortised |
| Entity overlap | Jaccard per node | $O(\|R\| \cdot \|Q\|)$ | $\|Q\|$ bounded by question length, not $N$ |
| Connectivity | `nx.subgraph + connected_components` | $O(\|R\| + \|E_R\|)$ | Subgraph is a *view*, no copy |
| Hub detection | `statistics.mean + stdev` | $O(N)$ | One pass over all degrees |
| Hub filtering | Scan retrieved nodes | $O(\|R\|)$ | Compare against pre-computed threshold |
| **Total per evaluation** | | $O(\|R\| \cdot \|Q\| + N)$ | Graph build amortised after first call |

**Amortised complexity** (after graph is cached): $O(|R| \cdot |Q|)$ — entity overlap dominates.

---

## Weight Calibration

Default weights ($\alpha = 0.4$, $\beta = 0.4$, $\gamma = 0.2$) encode the following priors:

- $\alpha = \beta$: Semantic relevance and structural coherence are treated as co-equal.
  Neither is sufficient alone.
- $\gamma < \alpha, \beta$: Hub noise is a correction signal, not a primary scorer.
- $\alpha + \beta - \gamma_{\max} = 0.6$: Score range under maximum hub penalty remains meaningful.

**Custom weights** are accepted at construction time and echoed in the evaluation contract:

```python
from src.evaluation.graph_context_relevance import GraphContextRelevanceEvaluator
from src.utils.graph_store import SQLiteGraphStore

store = SQLiteGraphStore("outputs/my_run/kg.db")
evaluator = GraphContextRelevanceEvaluator(store, alpha=0.5, beta=0.3, gamma=0.2)
result = evaluator.evaluate(
    question="What defects appear on the steel surface?",
    expected_answer="Scratches and pits are detected.",
    retrieved_node_hashes=["abc123...", "def456..."],
)
print(result["score"])        # float in [0.0, 1.0]
print(result["contract"])     # full diagnostic breakdown
```

**Domain calibration procedure:**
1. Collect human-labeled "good retrieval" / "bad retrieval" pairs from the domain corpus.
2. Grid-search $(\alpha, \beta, \gamma)$ subject to $\alpha, \beta, \gamma \geq 0$ and $\alpha + \beta > \gamma$.
3. Maximise rank correlation (Kendall-$\tau$) between GCR scores and human judgments.
4. Update `pipeline_config.yaml` `gcr_weights` section with calibrated values.

---

## Edge-Case Behaviour

| Input | $S_e$ | $S_c$ | $P_h$ | GCR | Notes |
|-------|-------|-------|-------|-----|-------|
| Empty retrieved set | 0.0 | 0.0 | 0.0 | **0.0** | Short-circuited before graph build |
| Single valid node | Jaccard score | **1.0** | 0 or 1 | varies | Single-node connectivity = 1 |
| All nodes irrelevant (tokens disjoint) | **0.0** | varies | varies | depends on $S_c$ | $S_e = 0$ pulls composite towards 0 |
| Node hashes not in store | — | — | — | Filtered; only valid hashes scored | `valid = [h for h in retrieved if h in full_graph]` |
| NaN content in node properties | 0.0 | — | — | Graceful; `_node_tokens` returns empty set | Tested in `test_nan_content_node_handled_gracefully` |
| Zero-variance degree distribution | — | — | **0.0** | No hub penalty | Regular graphs, e.g. $K_n$ |

---

## Two-Tier Entity Highlighter

The **QA Debugger** in the Insights Portal automatically highlights shared entities between the
question and retrieved context passages. Two strategies are applied in priority order:

### Tier 1: Explicit entity arrays (structured data)

Checks the `extra` payload of each QA item for explicit entity lists under any of these keys:

```
entities | extracted_entities | entity_list | named_entities
```

If found, those strings are used directly as highlight terms. This tier is **exact and precise** —
it uses the entities extracted by the evaluation pipeline's NER step.

### Tier 2: Answer-token overlap (heuristic fallback)

When no explicit entities are available, tokenises the LLM-generated answer and uses the
**unique content words** (length ≥ 4 chars) as highlight terms. Terms are:
- De-duplicated
- Sorted longest-first (greedy matching priority)
- Capped at 12 terms

```typescript
// Scoring heuristic — from insights-portal/src/utils/textHighlighter.ts
const words = answer
  .split(/[\s,.;:!?、。，！？·\-–—/\\()[\]{}"「」『』【】《》〈〉]+/)
  .filter((w) => w.length >= 4)
return [...new Set(words)].sort((a, b) => b.length - a.length).slice(0, 12)
```

### Security note

All highlight markup uses a hardcoded `<mark class="hl-entity">$1</mark>` / `<mark class="hl-overlap">$1</mark>` template. The `term` strings are regex-escaped before insertion. No user-controlled HTML is injected — the input data comes from the user's own locally loaded evaluation CSV files.

---

## Output Contract

Every `evaluate()` call returns a dict with the following guaranteed structure:

```python
{
    "score": float,          # Composite GCR in [0.0, 1.0]
    "contract": {
        "backend":                  "graph_context_relevance",
        "entity_overlap":           float,   # S_e in [0.0, 1.0]
        "structural_connectivity":  float,   # S_c in [0.0, 1.0]
        "hub_noise_penalty":        float,   # P_h in [0.0, 1.0]
        "hub_nodes":                list[str],  # hashes of flagged hub nodes
        "largest_component_size":   int,
        "retrieved_count":          int,     # len(valid hashes)
        "alpha":                    float,
        "beta":                     float,
        "gamma":                    float,
    }
}
```

All float values are rounded to 6 decimal places.

---

## Related Files

| File | Role |
|------|------|
| `eval-pipeline/src/evaluation/graph_context_relevance.py` | GCR evaluator implementation |
| `eval-pipeline/src/utils/graph_store.py` | `GraphStore` protocol + `SQLiteGraphStore` backend |
| `eval-pipeline/tests/test_graph_context_relevance.py` | 5-fixture TDD test suite |
| `eval-pipeline/tests/test_graph_store.py` | GraphStore unit tests |
| `insights-portal/src/utils/textHighlighter.ts` | Two-tier entity highlighter |
| `insights-portal/src/core/insights/engine.ts` | Deterministic rule-based insights engine |
| `config/pipeline_config.yaml` | `gcr_weights` and threshold configuration |
