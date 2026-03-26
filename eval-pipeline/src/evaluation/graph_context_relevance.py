"""Graph-based Context Relevance evaluator.

This module implements :class:`GraphContextRelevanceEvaluator`, a fully
offline, deterministic metric that evaluates the **topological quality** of a
retrieved subgraph against a question/answer pair.

Algorithm
---------
The composite score is defined as::

    GCR = clip(α·Sₑ + β·Sc − γ·Ph, 0.0, 1.0)

where:

* **Sₑ** (Entity Overlap, ``[0, 1]``) — Jaccard similarity between the token
  set derived from the question + expected answer and the per-node token sets
  (keyphrases + content).  Measures semantic relevance.

* **Sc** (Structural Connectivity, ``[0, 1]``) — ratio of the largest connected
  component to the total number of retrieved nodes, computed over an undirected
  projection of the relationships whose *both* endpoints are in the retrieved
  set.  Rewards coherent retrieval, penalises fragmented results.

* **Ph** (Hub Noise Penalty, ``[0, 1]``) — fraction of retrieved nodes whose
  degree in the *full* graph exceeds μ + 2σ.  Discourages retrieving massive
  "hub" nodes that dilute the context.

Default weights: α = 0.4, β = 0.4, γ = 0.2.

Complexity
----------
* Build full NetworkX graph: O(N + E) where N = total nodes, E = total edges
* Entity overlap: O(|R| · |Q|) per evaluation
* Connectivity: O(|R| + |E_R|) via BFS/DFS inside networkx
* Hub detection: O(N) for degree statistics, O(|R|) for filtering

All operations are offline/deterministic — no model calls, no network I/O.
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

import networkx as nx

from src.utils.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> frozenset[str]:
    """Lowercase alphanumeric token set from *text*."""
    return frozenset(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", text.lower()))


def _node_tokens(properties: Dict[str, Any]) -> frozenset[str]:
    """Aggregate tokens from a node's keyphrases, entities, and content fields."""
    parts: list[str] = []
    for field in ("keyphrases", "entities", "content", "page_content", "text"):
        val = properties.get(field)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend(str(v) for v in val)
    return _tokenize(" ".join(parts))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class GraphContextRelevanceEvaluator:
    """Evaluate retrieved-subgraph quality using topological graph metrics.

    Parameters
    ----------
    store:
        A :class:`~src.utils.graph_store.GraphStore` implementation
        (e.g. :class:`~src.utils.graph_store.SQLiteGraphStore`).
    alpha:
        Weight for the entity overlap component (default 0.4).
    beta:
        Weight for the structural connectivity component (default 0.4).
    gamma:
        Weight for the hub-noise penalty (default 0.2).

    Notes
    -----
    The full ``networkx`` graph is built lazily on the first call to
    :meth:`evaluate` and cached for subsequent calls within the same instance.
    Invalidate the cache by calling :meth:`invalidate_cache`.
    """

    def __init__(
        self,
        store: GraphStore,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
    ) -> None:
        if abs(alpha + beta - gamma - (alpha + beta - gamma)) > 1e-9:
            pass  # no strict normalisation required; clip handles overflow
        self._store = store
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._graph: Optional[nx.Graph] = None
        self._node_props: Optional[Dict[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Force a rebuild of the internal networkx graph on the next call."""
        self._graph = None
        self._node_props = None

    def _build_graph(self) -> tuple[nx.Graph, Dict[str, Dict[str, Any]]]:
        """Build (or return cached) full-graph + node-properties index.

        Returns
        -------
        (graph, node_props)
            ``graph`` is an undirected :class:`networkx.Graph` of all nodes
            and relationships from the store.
            ``node_props`` maps node_hash → properties dict.
        """
        if self._graph is not None and self._node_props is not None:
            return self._graph, self._node_props

        g: nx.Graph = nx.Graph()
        props: Dict[str, Dict[str, Any]] = {}

        for node in self._store.get_all_nodes():
            h = node["node_hash"]
            g.add_node(h)
            props[h] = node["properties"]

        for rel in self._store.get_all_relationships():
            # Only add edges between nodes that exist (guard against orphaned rels)
            if rel["src_hash"] in g and rel["tgt_hash"] in g:
                g.add_edge(rel["src_hash"], rel["tgt_hash"])

        self._graph = g
        self._node_props = props
        return g, props

    # ------------------------------------------------------------------
    # Sub-score calculations
    # ------------------------------------------------------------------

    def _entity_overlap(
        self,
        query_tokens: frozenset[str],
        retrieved: List[str],
        props: Dict[str, Dict[str, Any]],
    ) -> float:
        """Sₑ — mean Jaccard similarity between Q/A tokens and retrieved node tokens.

        Complexity: O(|R| · |Q|)
        """
        if not retrieved:
            return 0.0
        total = 0.0
        for h in retrieved:
            node_toks = _node_tokens(props.get(h, {}))
            total += _jaccard(query_tokens, node_toks)
        return total / len(retrieved)

    def _structural_connectivity(
        self,
        retrieved: List[str],
        full_graph: nx.Graph,
    ) -> float:
        """Sc — largest-component fraction of the retrieved subgraph.

        Complexity: O(|R| + |E_R|)
        """
        if not retrieved:
            return 0.0
        if len(retrieved) == 1:
            return 1.0
        subg = full_graph.subgraph(retrieved)
        largest_cc = max(nx.connected_components(subg), key=len, default=set())
        return len(largest_cc) / len(retrieved)

    def _hub_noise_penalty(
        self,
        retrieved: List[str],
        full_graph: nx.Graph,
    ) -> float:
        """Ph — fraction of retrieved nodes classified as degree hubs.

        A node is a *hub* if its degree in the full graph exceeds μ + 2σ.
        Returns 0.0 when the graph has fewer than 2 nodes or zero degree variance.

        Complexity: O(N) for stats, O(|R|) for filtering.
        """
        if not retrieved:
            return 0.0
        degrees = [d for _, d in full_graph.degree()]
        if len(degrees) < 2:
            return 0.0
        mean_deg = statistics.mean(degrees)
        try:
            stdev_deg = statistics.stdev(degrees)
        except statistics.StatisticsError:
            return 0.0
        if stdev_deg == 0.0:
            return 0.0
        threshold = mean_deg + 2.0 * stdev_deg
        hub_count = sum(
            1 for h in retrieved if full_graph.degree(h) > threshold
        )
        return hub_count / len(retrieved)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        retrieved_node_hashes: List[str],
    ) -> Dict[str, Any]:
        """Compute the Graph Context Relevance score for a retrieval result.

        Parameters
        ----------
        question:
            The user query string.
        expected_answer:
            The gold / reference answer.
        retrieved_node_hashes:
            Ordered list of node content-hashes representing the retrieved
            context subgraph.

        Returns
        -------
        Dict[str, Any]
            A result dict with keys:

            * ``"score"`` — composite GCR score in ``[0.0, 1.0]``.
            * ``"contract"`` — sub-score breakdown and diagnostic fields.

        Examples
        --------
        >>> evaluator.evaluate(
        ...     question="What defects appear on the steel surface?",
        ...     expected_answer="Scratches and pits are detected.",
        ...     retrieved_node_hashes=["abc123...", "def456..."],
        ... )
        {"score": 0.72, "contract": {...}}
        """
        if not retrieved_node_hashes:
            return {
                "score": 0.0,
                "contract": {
                    "backend": "graph_context_relevance",
                    "entity_overlap": 0.0,
                    "structural_connectivity": 0.0,
                    "hub_noise_penalty": 0.0,
                    "hub_nodes": [],
                    "largest_component_size": 0,
                    "retrieved_count": 0,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                },
            }

        full_graph, props = self._build_graph()

        # Only operate on hashes that actually exist in the store
        valid = [h for h in retrieved_node_hashes if h in full_graph]

        query_tokens = _tokenize(f"{question} {expected_answer}")

        s_e = self._entity_overlap(query_tokens, valid, props)
        s_c = self._structural_connectivity(valid, full_graph)
        p_h = self._hub_noise_penalty(valid, full_graph)

        raw = self.alpha * s_e + self.beta * s_c - self.gamma * p_h
        score = round(max(0.0, min(1.0, raw)), 6)

        # Identify hub hashes for diagnostics
        degrees = dict(full_graph.degree())
        all_degs = list(degrees.values())
        hub_threshold = 0.0
        if len(all_degs) >= 2:
            mean_d = statistics.mean(all_degs)
            try:
                std_d = statistics.stdev(all_degs)
                hub_threshold = mean_d + 2.0 * std_d
            except statistics.StatisticsError:
                pass
        hub_nodes = [h for h in valid if degrees.get(h, 0) > hub_threshold]

        subg = full_graph.subgraph(valid)
        largest_cc_size = max(
            (len(c) for c in nx.connected_components(subg)), default=0
        )

        return {
            "score": score,
            "contract": {
                "backend": "graph_context_relevance",
                "entity_overlap": round(s_e, 6),
                "structural_connectivity": round(s_c, 6),
                "hub_noise_penalty": round(p_h, 6),
                "hub_nodes": hub_nodes,
                "largest_component_size": largest_cc_size,
                "retrieved_count": len(valid),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        }
