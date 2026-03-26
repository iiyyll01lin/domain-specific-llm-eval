"""Tests for services/kg/summary.py (TASK-063)."""
import pytest
from services.kg.summary import (
    build_kg_summary,
    compute_degree_histogram,
    get_top_entities,
)


def _make_nodes(n=4):
    entities_sets = [
        ["steel", "plate", "surface"],
        ["steel", "quality", "check"],
        ["conveyor", "belt", "speed"],
        ["steel", "conveyor", "robot"],
    ]
    return [
        {"node_id": f"doc{i}", "entities": entities_sets[i % len(entities_sets)]}
        for i in range(n)
    ]


def _make_rels(nodes):
    """Connect doc0–doc1 and doc1–doc2."""
    return [
        {"source": nodes[0]["node_id"], "target": nodes[1]["node_id"], "type": "jaccard_similarity", "score": 0.3},
        {"source": nodes[1]["node_id"], "target": nodes[2]["node_id"], "type": "overlap_score", "score": 0.2},
    ] if len(nodes) >= 3 else []


def test_compute_degree_histogram_returns_list():
    nodes = _make_nodes()
    rels = _make_rels(nodes)
    hist = compute_degree_histogram(nodes, rels)
    assert isinstance(hist, list)


def test_compute_degree_histogram_max_bins():
    nodes = _make_nodes(10)
    rels = [
        {"source": nodes[i]["node_id"], "target": nodes[(i + 1) % 10]["node_id"]}
        for i in range(10)
    ]
    hist = compute_degree_histogram(nodes, rels, max_bins=50)
    assert len(hist) <= 50


def test_compute_degree_histogram_empty_nodes():
    assert compute_degree_histogram([], []) == []


def test_compute_degree_histogram_bin_fields():
    nodes = _make_nodes(4)
    rels = _make_rels(nodes)
    hist = compute_degree_histogram(nodes, rels)
    for bin_ in hist:
        assert "bin_start" in bin_
        assert "bin_end" in bin_
        assert "count" in bin_


def test_get_top_entities_returns_limit():
    nodes = _make_nodes(4)
    top = get_top_entities(nodes, limit=2)
    assert len(top) <= 2


def test_get_top_entities_fields():
    nodes = _make_nodes(4)
    top = get_top_entities(nodes)
    for item in top:
        assert "entity" in item
        assert "count" in item


def test_get_top_entities_sorted_by_frequency():
    nodes = _make_nodes(4)
    # "steel" appears in doc0, doc1, doc3 → highest frequency
    top = get_top_entities(nodes, limit=5)
    if top:
        top_entity_names = [t["entity"] for t in top]
        # "steel" has count 3, should appear first
        assert top[0]["entity"] == "steel"


def test_build_kg_summary_fields():
    nodes = _make_nodes(4)
    rels = _make_rels(nodes)
    summary = build_kg_summary(nodes, rels, kg_id="test-kg-123")
    assert summary["kg_id"] == "test-kg-123"
    assert summary["node_count"] == 4
    assert summary["edge_count"] == len(rels)
    assert "top_entities" in summary
    assert "degree_histogram" in summary
    assert "histogram_bins" in summary


def test_build_kg_summary_histogram_bins_limit():
    nodes = _make_nodes(4)
    rels = _make_rels(nodes)
    summary = build_kg_summary(nodes, rels, kg_id="x")
    assert summary["histogram_bins"] <= 50


def test_build_kg_summary_top_entity_limit():
    nodes = _make_nodes(4)
    rels = _make_rels(nodes)
    summary = build_kg_summary(nodes, rels, kg_id="x", top_entity_limit=2)
    assert len(summary["top_entities"]) <= 2
