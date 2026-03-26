"""Tests for services/kg/subgraph.py (TASK-066)."""
import pytest
from services.kg.subgraph import sample_subgraph


def _make_nodes(n=6):
    return [{"node_id": f"doc{i}", "entities": [f"entity{i}"]} for i in range(n)]


def _make_chain_rels(nodes):
    """Chain: doc0–doc1–doc2–doc3–…"""
    return [
        {
            "source": nodes[i]["node_id"],
            "target": nodes[i + 1]["node_id"],
            "type": "jaccard_similarity",
            "score": 0.5,
        }
        for i in range(len(nodes) - 1)
    ]


def test_sample_subgraph_empty_nodes():
    result = sample_subgraph([], [], seed_entity=None)
    assert result["nodes"] == []
    assert result["relationships"] == []
    assert result["truncated"] is False
    assert result["seed_node_id"] is None


def test_sample_subgraph_returns_required_keys():
    nodes = _make_nodes(3)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels)
    for key in ("nodes", "relationships", "truncated", "seed_node_id"):
        assert key in result


def test_sample_subgraph_not_truncated_small_graph():
    nodes = _make_nodes(3)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels, max_nodes=500)
    assert result["truncated"] is False
    assert len(result["nodes"]) == 3


def test_sample_subgraph_truncated_when_max_exceeded():
    nodes = _make_nodes(6)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels, max_nodes=3)
    assert result["truncated"] is True
    assert len(result["nodes"]) == 3


def test_sample_subgraph_deterministic():
    nodes = _make_nodes(6)
    rels = _make_chain_rels(nodes)
    r1 = sample_subgraph(nodes, rels, max_nodes=3, seed=42)
    r2 = sample_subgraph(nodes, rels, max_nodes=3, seed=42)
    assert [n["node_id"] for n in r1["nodes"]] == [n["node_id"] for n in r2["nodes"]]


def test_sample_subgraph_seed_entity_resolves():
    nodes = _make_nodes(4)
    rels = _make_chain_rels(nodes)
    # doc2 has entity "entity2"
    result = sample_subgraph(nodes, rels, seed_entity="entity2")
    assert result["seed_node_id"] == "doc2"


def test_sample_subgraph_seed_node_id_direct():
    nodes = _make_nodes(4)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels, seed_entity="doc3")
    assert result["seed_node_id"] == "doc3"


def test_sample_subgraph_relationships_subset():
    nodes = _make_nodes(6)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels, max_nodes=3)
    selected_ids = {n["node_id"] for n in result["nodes"]}
    for rel in result["relationships"]:
        assert rel["source"] in selected_ids
        assert rel["target"] in selected_ids


def test_sample_subgraph_unknown_seed_falls_back():
    nodes = _make_nodes(3)
    rels = _make_chain_rels(nodes)
    result = sample_subgraph(nodes, rels, seed_entity="nonexistent_entity")
    assert result["seed_node_id"] is not None
    assert result["seed_node_id"] in {n["node_id"] for n in nodes}
