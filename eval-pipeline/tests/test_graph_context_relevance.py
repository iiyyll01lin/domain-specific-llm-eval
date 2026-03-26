"""TDD tests for GraphContextRelevanceEvaluator.

Fixtures
--------
Three synthetic graph topologies are constructed using SQLiteGraphStore so
tests exercise the real persistence layer (in-memory `:memory:` path) and the
real networkx graph-building logic:

1. **perfect_store** — A 4-node chain (A–B–C–D) where every node is highly
   relevant, all nodes are connected in a single component, and no hub nodes
   exist.  Expected: high score (≥ 0.5).

2. **fragmented_store** — 4 nodes, zero edges.  Node content is relevant but
   the subgraph is fully disconnected.  Expected: lower score than perfect due
   to structural_connectivity = 0.25 (each node its own component).

3. **hub_store** — 3 relevant "spoke" nodes all connected to a single massive
   hub node that has 10 extra cross-edges to irrelevant nodes.  The hub's
   degree >> μ + 2σ so Ph > 0.  Expected: hub noise reduces the final score
   compared to the all-relevant no-hub baseline.

4. **empty_retrieval** — No hashes retrieved: score must be exactly 0.0.

5. **single_node** — Single relevant node: structural_connectivity = 1.0.

Algorithm contract checks
--------------------------
* Score always in [0.0, 1.0].
* ``contract`` dict always present with required keys.
* Determinism: two identical calls return identical scores.
* Weight override: custom α/β/γ values are reflected in output.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.evaluation.graph_context_relevance import (
    GraphContextRelevanceEvaluator,
    _jaccard,
    _node_tokens,
    _tokenize,
)
from src.utils.graph_store import SQLiteGraphStore, hash_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTRACT_KEYS = {
    "backend",
    "entity_overlap",
    "structural_connectivity",
    "hub_noise_penalty",
    "hub_nodes",
    "largest_component_size",
    "retrieved_count",
    "alpha",
    "beta",
    "gamma",
}


def _make_node(
    store: SQLiteGraphStore,
    content: str,
    node_type: str = "document",
    extra_props: Dict[str, Any] | None = None,
) -> str:
    """Upsert a node and return its hash."""
    h = hash_content(content)
    props: Dict[str, Any] = {"content": content, "keyphrases": content}
    if extra_props:
        props.update(extra_props)
    store.upsert_node(h, node_type, props)
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def perfect_store(tmp_path: Path) -> tuple[SQLiteGraphStore, List[str]]:
    """4-node chain A–B–C–D, all relevant to 'steel surface defect inspection'."""
    store = SQLiteGraphStore(tmp_path / "perfect.db")
    ha = _make_node(store, "steel surface defect inspection scratch pit")
    hb = _make_node(store, "surface quality inspection defect detection")
    hc = _make_node(store, "steel plate scratch detection method")
    hd = _make_node(store, "pit defect surface steel quality")
    store.add_relationship(ha, hb, "jaccard_similarity", {"score": 0.8})
    store.add_relationship(hb, hc, "jaccard_similarity", {"score": 0.75})
    store.add_relationship(hc, hd, "jaccard_similarity", {"score": 0.7})
    return store, [ha, hb, hc, hd]


@pytest.fixture()
def fragmented_store(tmp_path: Path) -> tuple[SQLiteGraphStore, List[str]]:
    """4 relevant nodes with NO relationships (fully disconnected)."""
    store = SQLiteGraphStore(tmp_path / "fragmented.db")
    ha = _make_node(store, "steel surface defect inspection")
    hb = _make_node(store, "surface quality inspection scratch")
    hc = _make_node(store, "steel plate pit detection")
    hd = _make_node(store, "defect surface steel quality method")
    return store, [ha, hb, hc, hd]


@pytest.fixture()
def hub_store(tmp_path: Path) -> tuple[SQLiteGraphStore, List[str], str]:
    """3 relevant spokes (triangle-connected) + 1 massive hub with extra noise edges.

    The spokes form a fully-connected triangle so they can achieve S_c = 1.0
    *without* the hub.  When the hub is included, S_c remains 1.0 but Ph > 0
    (hub degree >> μ + 2σ), making the hub-included score strictly lower.

    Returns (store, [h1, h2, h3, hub], hub_hash).
    """
    store = SQLiteGraphStore(tmp_path / "hub.db")
    # Relevant spokes — triangle connectivity so spokes_only is fully connected
    h1 = _make_node(store, "steel defect surface scratch")
    h2 = _make_node(store, "pit detection steel quality")
    h3 = _make_node(store, "surface inspection defect")
    store.add_relationship(h1, h2, "jaccard_similarity", {"score": 0.65})
    store.add_relationship(h2, h3, "jaccard_similarity", {"score": 0.65})
    store.add_relationship(h1, h3, "jaccard_similarity", {"score": 0.65})
    # Hub node (relevant content but massively over-connected)
    hub = _make_node(store, "steel surface defect inspection generic hub node", node_type="hub")
    # Connect each spoke to hub (hub degree grows)
    store.add_relationship(h1, hub, "jaccard_similarity", {"score": 0.6})
    store.add_relationship(h2, hub, "jaccard_similarity", {"score": 0.6})
    store.add_relationship(h3, hub, "jaccard_similarity", {"score": 0.6})
    # Add ~10 extra irrelevant noise nodes connected only to the hub to inflate its degree
    for i in range(10):
        hn = _make_node(store, f"irrelevant random content node {i} unrelated topic")
        store.add_relationship(hub, hn, "jaccard_similarity", {"score": 0.1})
    return store, [h1, h2, h3, hub], hub


@pytest.fixture()
def single_node_store(tmp_path: Path) -> tuple[SQLiteGraphStore, str]:
    """A store with exactly one relevant node."""
    store = SQLiteGraphStore(tmp_path / "single.db")
    h = _make_node(store, "steel surface defect inspection")
    return store, h


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_tokenize_lowercases(self) -> None:
        tokens = _tokenize("Steel Surface DEFECT")
        assert "steel" in tokens
        assert "surface" in tokens
        assert "defect" in tokens

    def test_tokenize_strips_punctuation(self) -> None:
        tokens = _tokenize("defect, scratch! pit.")
        assert "defect" in tokens
        assert "scratch" in tokens
        assert "pit" in tokens
        assert "," not in tokens

    def test_tokenize_handles_chinese(self) -> None:
        tokens = _tokenize("鋼板表面缺陷")
        assert len(tokens) > 0

    def test_jaccard_identical_sets(self) -> None:
        a = frozenset(["a", "b", "c"])
        assert _jaccard(a, a) == 1.0

    def test_jaccard_disjoint_sets(self) -> None:
        a = frozenset(["x", "y"])
        b = frozenset(["a", "b"])
        assert _jaccard(a, b) == 0.0

    def test_jaccard_partial_overlap(self) -> None:
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        result = _jaccard(a, b)
        assert abs(result - 2 / 4) < 1e-9  # |{b,c}| / |{a,b,c,d}|

    def test_jaccard_empty_set_returns_zero(self) -> None:
        assert _jaccard(frozenset(), frozenset(["a"])) == 0.0
        assert _jaccard(frozenset(["a"]), frozenset()) == 0.0

    def test_node_tokens_aggregates_fields(self) -> None:
        props = {
            "keyphrases": "steel defect",
            "content": "scratch pit",
            "entities": ["surface", "quality"],
        }
        tokens = _node_tokens(props)
        assert "steel" in tokens
        assert "scratch" in tokens
        assert "surface" in tokens

    def test_node_tokens_missing_fields_returns_empty(self) -> None:
        tokens = _node_tokens({})
        assert isinstance(tokens, frozenset)
        assert len(tokens) == 0


# ---------------------------------------------------------------------------
# Contract / edge case tests (store-agnostic)
# ---------------------------------------------------------------------------

class TestGraphContextRelevanceContracts:
    """Score and contract structural guarantees — must hold for all topologies."""

    def test_empty_retrieval_returns_zero(self, perfect_store: tuple) -> None:
        store, _ = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="What are steel surface defects?",
            expected_answer="Scratches and pits.",
            retrieved_node_hashes=[],
        )
        assert result["score"] == 0.0
        assert CONTRACT_KEYS.issubset(result["contract"].keys())

    def test_score_is_in_unit_interval(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel surface defect", expected_answer="scratch pit", retrieved_node_hashes=hashes
        )
        assert 0.0 <= result["score"] <= 1.0

    def test_contract_keys_present(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel surface defect", expected_answer="scratch pit", retrieved_node_hashes=hashes
        )
        assert CONTRACT_KEYS.issubset(result["contract"].keys())
        assert result["contract"]["backend"] == "graph_context_relevance"

    def test_determinism(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        r1 = evaluator.evaluate("steel defect?", "scratch pit", hashes)
        r2 = evaluator.evaluate("steel defect?", "scratch pit", hashes)
        assert r1["score"] == r2["score"]
        assert r1["contract"] == r2["contract"]

    def test_weight_override_reflected_in_contract(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store, alpha=0.6, beta=0.3, gamma=0.1)
        result = evaluator.evaluate("steel defect?", "scratch pit", hashes)
        assert result["contract"]["alpha"] == 0.6
        assert result["contract"]["beta"] == 0.3
        assert result["contract"]["gamma"] == 0.1

    def test_retrieved_count_matches_valid_hashes(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("steel defect?", "scratch", hashes)
        assert result["contract"]["retrieved_count"] == len(hashes)

    def test_unknown_hashes_are_silently_excluded(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        with_ghost = hashes + ["nonexistent_ghost_hash_000"]
        result = evaluator.evaluate("steel defect?", "scratch", with_ghost)
        # Only valid hashes contribute
        assert result["contract"]["retrieved_count"] == len(hashes)


# ---------------------------------------------------------------------------
# Topology-specific score correctness
# ---------------------------------------------------------------------------

class TestPerfectSubgraph:
    """A fully connected chain of relevant nodes should score highly."""

    def test_perfect_subgraph_scores_high(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel surface defect inspection",
            expected_answer="scratch pit detection on steel plate",
            retrieved_node_hashes=hashes,
        )
        # Entity overlap is high (all nodes are relevant) + connectivity = 1.0
        assert result["score"] >= 0.5, f"Expected ≥ 0.5, got {result['score']}"

    def test_perfect_subgraph_structural_connectivity_is_one(self, perfect_store: tuple) -> None:
        """Chain A–B–C–D is a single component → Sc = 1.0."""
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel defect", expected_answer="scratch pit", retrieved_node_hashes=hashes
        )
        assert result["contract"]["structural_connectivity"] == 1.0

    def test_perfect_subgraph_no_hub_nodes(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="defect inspection", expected_answer="scratch pit", retrieved_node_hashes=hashes
        )
        # In a 4-node chain, all degrees are ≤ 2; no hubs
        assert result["contract"]["hub_nodes"] == []
        assert result["contract"]["hub_noise_penalty"] == 0.0


class TestFragmentedSubgraph:
    """Zero edges → structural_connectivity should be 1/|R| (each its own component)."""

    def test_fragmented_connectivity_is_one_over_n(self, fragmented_store: tuple) -> None:
        store, hashes = fragmented_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel defect", expected_answer="scratch pit", retrieved_node_hashes=hashes
        )
        expected_sc = 1.0 / len(hashes)  # largest CC = 1 node
        assert abs(result["contract"]["structural_connectivity"] - expected_sc) < 1e-9

    def test_fragmented_scores_lower_than_perfect(
        self,
        perfect_store: tuple,
        fragmented_store: tuple,
    ) -> None:
        """Same number of relevant nodes; perfect (connected) > fragmented (disconnected)."""
        q = "steel surface defect inspection"
        a = "scratch pit detection steel plate"
        s_perfect, hashes_p = perfect_store
        s_frag, hashes_f = fragmented_store

        ev_p = GraphContextRelevanceEvaluator(s_perfect)
        ev_f = GraphContextRelevanceEvaluator(s_frag)

        score_p = ev_p.evaluate(q, a, hashes_p)["score"]
        score_f = ev_f.evaluate(q, a, hashes_f)["score"]

        assert score_p > score_f, (
            f"Perfect connected score ({score_p}) should exceed "
            f"fragmented score ({score_f})"
        )

    def test_fragmented_largest_component_is_one(self, fragmented_store: tuple) -> None:
        store, hashes = fragmented_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("defect", "scratch", hashes)
        assert result["contract"]["largest_component_size"] == 1


class TestHubNoiseSubgraph:
    """A massive hub node should be identified and penalised."""

    def test_hub_node_is_detected(self, hub_store: tuple) -> None:
        store, spokes_and_hub, hub_hash = hub_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel surface defect inspection",
            expected_answer="scratch pit detection quality",
            retrieved_node_hashes=spokes_and_hub,
        )
        assert hub_hash in result["contract"]["hub_nodes"], (
            "The massive hub node should be flagged"
        )

    def test_hub_penalty_is_positive(self, hub_store: tuple) -> None:
        store, spokes_and_hub, _ = hub_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate(
            question="steel surface defect",
            expected_answer="scratch pit",
            retrieved_node_hashes=spokes_and_hub,
        )
        assert result["contract"]["hub_noise_penalty"] > 0.0

    def test_hub_penalty_reduces_score_vs_no_hub(self, hub_store: tuple, tmp_path: Path) -> None:
        """Score with hub included should be ≤ score without hub (spokes only)."""
        store, spokes_and_hub, hub_hash = hub_store
        spokes_only = [h for h in spokes_and_hub if h != hub_hash]

        evaluator = GraphContextRelevanceEvaluator(store)
        evaluator.invalidate_cache()

        q = "steel surface defect inspection"
        a = "scratch pit detection quality"

        score_with_hub = evaluator.evaluate(q, a, spokes_and_hub)["score"]
        evaluator.invalidate_cache()
        score_spokes_only = evaluator.evaluate(q, a, spokes_only)["score"]

        # Including the hub should not improve the score (penalty applies)
        assert score_with_hub <= score_spokes_only + 1e-9, (
            f"Hub-included score {score_with_hub} should be ≤ spokes-only {score_spokes_only}"
        )


class TestSingleNode:
    """Single-node retrieval: structural_connectivity = 1.0 by definition."""

    def test_single_node_connectivity_is_one(self, single_node_store: tuple) -> None:
        store, h = single_node_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("steel defect?", "scratch", [h])
        assert result["contract"]["structural_connectivity"] == 1.0

    def test_single_node_score_in_unit_interval(self, single_node_store: tuple) -> None:
        store, h = single_node_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("steel surface defect", "scratch pit", [h])
        assert 0.0 <= result["score"] <= 1.0

    def test_single_node_no_hub_penalty(self, single_node_store: tuple) -> None:
        store, h = single_node_store
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("steel defect", "scratch", [h])
        # With only 1 node, degree std < 2 → no hub threshold → no penalty
        assert result["contract"]["hub_noise_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def test_invalidate_cache_forces_rebuild(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        r1 = evaluator.evaluate("steel", "scratch", hashes)
        evaluator.invalidate_cache()
        r2 = evaluator.evaluate("steel", "scratch", hashes)
        # Results should be identical after cache rebuild
        assert r1["score"] == r2["score"]

    def test_graph_is_lazily_built(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        assert evaluator._graph is None  # not yet built
        evaluator.evaluate("steel", "scratch", hashes)
        assert evaluator._graph is not None  # built after first call

    def test_second_call_uses_cached_graph(self, perfect_store: tuple) -> None:
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store)
        evaluator.evaluate("steel", "scratch", hashes)
        graph_id_first = id(evaluator._graph)
        evaluator.evaluate("steel", "scratch", hashes)
        graph_id_second = id(evaluator._graph)
        assert graph_id_first == graph_id_second


# ---------------------------------------------------------------------------
# Numerical boundary tests
# ---------------------------------------------------------------------------

class TestNumericalBoundaries:
    def test_score_never_below_zero(self, hub_store: tuple) -> None:
        """Even with maximum penalty, score must not go negative."""
        store, hashes, _ = hub_store
        # Cranking up gamma to an extreme value
        evaluator = GraphContextRelevanceEvaluator(store, alpha=0.1, beta=0.1, gamma=5.0)
        result = evaluator.evaluate("steel", "scratch", hashes)
        assert result["score"] >= 0.0

    def test_score_never_above_one(self, perfect_store: tuple) -> None:
        """With extreme alpha + beta, score must not exceed 1.0."""
        store, hashes = perfect_store
        evaluator = GraphContextRelevanceEvaluator(store, alpha=5.0, beta=5.0, gamma=0.0)
        result = evaluator.evaluate("steel surface defect inspection scratch pit", "scratch pit", hashes)
        assert result["score"] <= 1.0

    def test_nan_content_node_handled_gracefully(self, tmp_path: Path) -> None:
        """Nodes with empty/missing content should not crash the evaluator."""
        store = SQLiteGraphStore(tmp_path / "nan.db")
        h = hash_content("empty node")
        store.upsert_node(h, "document", {})  # no content field
        evaluator = GraphContextRelevanceEvaluator(store)
        result = evaluator.evaluate("question", "answer", [h])
        assert 0.0 <= result["score"] <= 1.0
