"""Tests for services/kg/relationships.py (TASK-062b/c)."""
import pytest
from services.kg.relationships import (
    build_all_relationships,
    build_cosine_relationships,
    build_jaccard_relationships,
    build_overlap_relationships,
    count_by_type,
)


def _make_nodes():
    return [
        {
            "node_id": "doc1",
            "entities": ["steel", "plate", "surface"],
            "keyphrases": ["steel plate", "surface inspection"],
        },
        {
            "node_id": "doc2",
            "entities": ["steel", "quality", "check"],
            "keyphrases": ["steel plate", "quality check"],
        },
        {
            "node_id": "doc3",
            "entities": ["conveyor", "belt", "speed"],
            "keyphrases": ["conveyor belt", "speed control"],
        },
    ]


def test_jaccard_finds_related_nodes():
    nodes = _make_nodes()
    rels = build_jaccard_relationships(nodes, threshold=0.1)
    # doc1 and doc2 share "steel" → jaccard = 1/5 = 0.2 ≥ 0.1
    assert any(
        (r["source"] == "doc1" and r["target"] == "doc2") or
        (r["source"] == "doc2" and r["target"] == "doc1")
        for r in rels
    )


def test_jaccard_type_label():
    nodes = _make_nodes()
    rels = build_jaccard_relationships(nodes, threshold=0.0)
    assert all(r["type"] == "jaccard_similarity" for r in rels)


def test_jaccard_score_in_range():
    nodes = _make_nodes()
    rels = build_jaccard_relationships(nodes, threshold=0.0)
    for r in rels:
        assert 0.0 <= r["score"] <= 1.0


def test_jaccard_threshold_filters():
    nodes = _make_nodes()
    rels_all = build_jaccard_relationships(nodes, threshold=0.0)
    rels_strict = build_jaccard_relationships(nodes, threshold=0.99)
    assert len(rels_strict) <= len(rels_all)


def test_overlap_finds_keyphrase_matches():
    nodes = _make_nodes()
    rels = build_overlap_relationships(nodes, threshold=0.05)
    # doc1 and doc2 share "steel plate"
    assert any(
        (r["source"] == "doc1" and r["target"] == "doc2") or
        (r["source"] == "doc2" and r["target"] == "doc1")
        for r in rels
    )


def test_overlap_type_label():
    nodes = _make_nodes()
    rels = build_overlap_relationships(nodes, threshold=0.0)
    assert all(r["type"] == "overlap_score" for r in rels)


def test_cosine_skips_nodes_without_embeddings():
    nodes = _make_nodes()
    rels, skipped = build_cosine_relationships(nodes, threshold=0.5)
    assert rels == []
    assert skipped == len(nodes)


def test_cosine_with_embeddings():
    nodes = [
        {"node_id": "a", "embedding": [1.0, 0.0, 0.0]},
        {"node_id": "b", "embedding": [1.0, 0.0, 0.0]},  # identical → cosine=1.0
        {"node_id": "c", "embedding": [0.0, 1.0, 0.0]},  # orthogonal → cosine=0.0
    ]
    rels, skipped = build_cosine_relationships(nodes, threshold=0.9)
    assert any(
        (r["source"] == "a" and r["target"] == "b") or
        (r["source"] == "b" and r["target"] == "a")
        for r in rels
    )
    # orthogonal pair should not appear
    assert not any(
        "c" in (r["source"], r["target"]) for r in rels
    )
    assert skipped == 0


def test_build_all_relationships_returns_list():
    nodes = _make_nodes()
    rels = build_all_relationships(nodes)
    assert isinstance(rels, list)


def test_build_all_low_threshold_returns_nonzero():
    nodes = _make_nodes()
    rels = build_all_relationships(
        nodes,
        jaccard_threshold=0.0,
        overlap_threshold=0.0,
        cosine_threshold=1.1,  # effectively disables cosine
        summary_cosine_threshold=1.1,
    )
    assert len(rels) > 0


def test_count_by_type():
    rels = [
        {"type": "jaccard_similarity", "score": 0.2},
        {"type": "jaccard_similarity", "score": 0.3},
        {"type": "overlap_score", "score": 0.5},
    ]
    counts = count_by_type(rels)
    assert counts["jaccard_similarity"] == 2
    assert counts["overlap_score"] == 1


def test_count_by_type_empty():
    assert count_by_type([]) == {}
