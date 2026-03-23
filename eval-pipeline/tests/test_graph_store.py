"""Unit tests for the GraphStore abstraction and SQLiteGraphStore implementation.

TDD — these tests were written BEFORE the implementation of graph_store.py.
They cover:
  - Content hashing (determinism, uniqueness, length)
  - SQLiteGraphStore CRUD (upsert, exists, dedup, list)
  - Relationship management (add, dedup, list)
  - Persistence across separate store instances (same db file)
  - Pruning of stale nodes
  - Incremental-build helpers (new-only node detection)
  - Neo4jGraphStore delegation to Neo4jGraphManager
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from src.utils.graph_store import (
    Neo4jGraphStore,
    NodeRecord,
    RelRecord,
    SQLiteGraphStore,
    hash_content,
)


# ---------------------------------------------------------------------------
# hash_content
# ---------------------------------------------------------------------------


def test_hash_content_is_deterministic() -> None:
    assert hash_content("hello world") == hash_content("hello world")


def test_hash_content_differs_for_different_content() -> None:
    assert hash_content("alpha") != hash_content("beta")


def test_hash_content_is_hex_string_of_length_32() -> None:
    h = hash_content("some document text")
    assert isinstance(h, str)
    assert len(h) == 32
    int(h, 16)  # raises ValueError if not valid hex


def test_hash_content_strips_surrounding_whitespace() -> None:
    """Leading/trailing whitespace should be ignored during hashing."""
    assert hash_content("  hello  ") == hash_content("hello")


# ---------------------------------------------------------------------------
# SQLiteGraphStore — basic construction
# ---------------------------------------------------------------------------


def test_sqlite_store_creates_db_file_on_init(tmp_path: Path) -> None:
    db_path = tmp_path / "kg.db"
    SQLiteGraphStore(db_path)
    assert db_path.exists()


def test_sqlite_store_schema_has_required_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "kg.db"
    SQLiteGraphStore(db_path)
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "nodes" in tables
    assert "relationships" in tables


# ---------------------------------------------------------------------------
# SQLiteGraphStore — upsert_node
# ---------------------------------------------------------------------------


def test_upsert_node_returns_true_for_new_node(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    node_hash = hash_content("chunk text alpha")
    is_new = store.upsert_node(node_hash, "document", {"content": "chunk text alpha", "title": "Doc A"})
    assert is_new is True


def test_upsert_node_returns_false_for_duplicate(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    node_hash = hash_content("chunk text beta")
    store.upsert_node(node_hash, "document", {"content": "chunk text beta"})
    is_new = store.upsert_node(node_hash, "document", {"content": "chunk text beta updated"})
    assert is_new is False


def test_upsert_node_updates_properties_on_duplicate(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    node_hash = hash_content("shared content")
    store.upsert_node(node_hash, "document", {"title": "v1", "score": 1})
    store.upsert_node(node_hash, "document", {"title": "v2", "score": 2})

    nodes = store.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0]["properties"]["title"] == "v2"
    assert nodes[0]["properties"]["score"] == 2


# ---------------------------------------------------------------------------
# SQLiteGraphStore — node_exists
# ---------------------------------------------------------------------------


def test_node_exists_returns_false_for_unknown_hash(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    assert store.node_exists("deadbeef" * 4) is False


def test_node_exists_returns_true_after_upsert(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h = hash_content("exists check")
    store.upsert_node(h, "document", {})
    assert store.node_exists(h) is True


# ---------------------------------------------------------------------------
# SQLiteGraphStore — get_all_nodes
# ---------------------------------------------------------------------------


def test_get_all_nodes_returns_empty_list_initially(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    assert store.get_all_nodes() == []


def test_get_all_nodes_returns_all_upserted_nodes(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("doc one")
    h2 = hash_content("doc two")
    store.upsert_node(h1, "document", {"title": "One"})
    store.upsert_node(h2, "document", {"title": "Two"})

    nodes = store.get_all_nodes()
    assert len(nodes) == 2
    hashes = {n["node_hash"] for n in nodes}
    assert hashes == {h1, h2}


def test_get_all_nodes_record_has_required_fields(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h = hash_content("field check doc")
    store.upsert_node(h, "document", {"content": "field check doc", "title": "FC"})

    node: NodeRecord = store.get_all_nodes()[0]
    assert node["node_hash"] == h
    assert node["node_type"] == "document"
    assert isinstance(node["properties"], dict)
    assert node["properties"]["title"] == "FC"
    assert "created_at" in node
    assert "updated_at" in node


# ---------------------------------------------------------------------------
# SQLiteGraphStore — persistent storage across instances
# ---------------------------------------------------------------------------


def test_store_persists_across_separate_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.db"
    h = hash_content("persistent node")

    store_a = SQLiteGraphStore(db_path)
    store_a.upsert_node(h, "document", {"title": "Persisted"})

    store_b = SQLiteGraphStore(db_path)
    assert store_b.node_exists(h) is True
    nodes = store_b.get_all_nodes()
    assert nodes[0]["properties"]["title"] == "Persisted"


# ---------------------------------------------------------------------------
# SQLiteGraphStore — add_relationship
# ---------------------------------------------------------------------------


def test_add_relationship_stores_edge(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("source node")
    h2 = hash_content("target node")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "jaccard_similarity", {"score": 0.42})

    rels = store.get_all_relationships()
    assert len(rels) == 1
    rel: RelRecord = rels[0]
    assert rel["src_hash"] == h1
    assert rel["tgt_hash"] == h2
    assert rel["rel_type"] == "jaccard_similarity"
    assert rel["properties"]["score"] == pytest.approx(0.42)


def test_add_relationship_is_idempotent(tmp_path: Path) -> None:
    """Same (src, tgt, rel_type) triple must not produce duplicate rows."""
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("idem src")
    h2 = hash_content("idem tgt")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "overlap_score", {"score": 0.1})
    store.add_relationship(h1, h2, "overlap_score", {"score": 0.2})

    rels = store.get_all_relationships()
    assert len(rels) == 1


def test_add_relationship_different_types_stored_independently(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("multi rel src")
    h2 = hash_content("multi rel tgt")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "jaccard_similarity", {"score": 0.3})
    store.add_relationship(h1, h2, "overlap_score", {"score": 0.5})

    rels = store.get_all_relationships()
    rel_types = {r["rel_type"] for r in rels}
    assert rel_types == {"jaccard_similarity", "overlap_score"}


# ---------------------------------------------------------------------------
# SQLiteGraphStore — get_all_relationships
# ---------------------------------------------------------------------------


def test_get_all_relationships_returns_empty_initially(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    assert store.get_all_relationships() == []


def test_get_all_relationships_record_has_required_fields(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("rel field src")
    h2 = hash_content("rel field tgt")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "cosine_similarity", {"cosine_similarity": 0.85})

    rel: RelRecord = store.get_all_relationships()[0]
    assert "src_hash" in rel
    assert "tgt_hash" in rel
    assert "rel_type" in rel
    assert "properties" in rel
    assert "created_at" in rel


# ---------------------------------------------------------------------------
# SQLiteGraphStore — prune_stale
# ---------------------------------------------------------------------------


def test_prune_stale_removes_old_nodes(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h_old = hash_content("old node")
    h_new = hash_content("new node")

    store.upsert_node(h_old, "document", {"title": "Old"})
    store.upsert_node(h_new, "document", {"title": "New"})

    # Manually back-date the old node's updated_at to force stale detection
    conn = sqlite3.connect(tmp_path / "kg.db")
    conn.execute(
        "UPDATE nodes SET updated_at = '2000-01-01T00:00:00' WHERE node_hash = ?",
        (h_old,),
    )
    conn.commit()
    conn.close()

    removed = store.prune_stale(max_age_days=1)
    assert removed == 1
    assert store.node_exists(h_new) is True
    assert store.node_exists(h_old) is False


def test_prune_stale_also_removes_dangling_relationships(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h1 = hash_content("prune cascade src")
    h2 = hash_content("prune cascade tgt")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "jaccard_similarity", {})

    # Back-date both nodes
    conn = sqlite3.connect(tmp_path / "kg.db")
    conn.execute("UPDATE nodes SET updated_at = '2000-01-01T00:00:00'")
    conn.commit()
    conn.close()

    store.prune_stale(max_age_days=1)
    assert store.get_all_relationships() == []


# ---------------------------------------------------------------------------
# SQLiteGraphStore — new-only node detection (incremental build helper)
# ---------------------------------------------------------------------------


def test_filter_new_hashes_returns_only_unseen_hashes(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h_existing = hash_content("already in store")
    h_new = hash_content("not yet in store")
    store.upsert_node(h_existing, "document", {})

    candidate_hashes = [h_existing, h_new]
    new_hashes = store.filter_new_hashes(candidate_hashes)
    assert new_hashes == [h_new]


def test_filter_new_hashes_returns_all_when_store_empty(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "kg.db")
    hashes = [hash_content("a"), hash_content("b")]
    assert store.filter_new_hashes(hashes) == hashes


# ---------------------------------------------------------------------------
# SQLiteGraphStore — node_uuid storage and retrieval
# ---------------------------------------------------------------------------


def test_upsert_node_stores_node_uuid_in_properties(tmp_path: Path) -> None:
    """_node_uuid must be stored so RAGAS Node IDs are stable across runs."""
    store = SQLiteGraphStore(tmp_path / "kg.db")
    node_uuid = str(uuid.uuid4())
    h = hash_content("uuid storage test")
    store.upsert_node(h, "document", {"content": "uuid storage test", "_node_uuid": node_uuid})

    nodes = store.get_all_nodes()
    assert nodes[0]["properties"]["_node_uuid"] == node_uuid


def test_upsert_node_auto_generates_node_uuid_when_absent(tmp_path: Path) -> None:
    """If caller omits _node_uuid, the store must synthesise one automatically."""
    store = SQLiteGraphStore(tmp_path / "kg.db")
    h = hash_content("auto uuid node")
    store.upsert_node(h, "document", {"content": "auto uuid node"})

    nodes = store.get_all_nodes()
    auto_uuid = nodes[0]["properties"].get("_node_uuid")
    assert auto_uuid is not None
    uuid.UUID(auto_uuid)  # must be a valid UUID string


# ---------------------------------------------------------------------------
# Neo4jGraphStore — delegation to Neo4jGraphManager
# ---------------------------------------------------------------------------


class _FakeNeo4jManager:
    """Lightweight stand-in for Neo4jGraphManager used to verify delegation."""

    def __init__(self) -> None:
        self.connected = False
        self.nodes: Dict[str, Any] = {}
        self.edges = []

    def connect(self) -> None:
        self.connected = True

    def add_node(self, node_id: str, **properties: Any) -> None:
        self.nodes[node_id] = {"id": node_id, **properties}

    def add_relationship(self, source: str, target: str, relation: str, **properties: Any) -> None:
        self.edges.append({"source": source, "target": target, "relation": relation, **properties})

    def execute_cypher(self, query: str):
        return []

    @property
    def backend(self) -> str:
        return "memory"


def test_neo4j_graph_store_upsert_delegates_to_manager(tmp_path: Path) -> None:
    fake_manager = _FakeNeo4jManager()
    store = Neo4jGraphStore(manager=fake_manager)

    h = hash_content("neo4j delegation test")
    store.upsert_node(h, "document", {"title": "Delegated"})

    assert h in fake_manager.nodes
    assert fake_manager.nodes[h]["title"] == "Delegated"


def test_neo4j_graph_store_add_relationship_delegates(tmp_path: Path) -> None:
    fake_manager = _FakeNeo4jManager()
    store = Neo4jGraphStore(manager=fake_manager)

    h1 = hash_content("neo4j src")
    h2 = hash_content("neo4j tgt")
    store.upsert_node(h1, "document", {})
    store.upsert_node(h2, "document", {})
    store.add_relationship(h1, h2, "overlap_score", {"score": 0.6})

    assert len(fake_manager.edges) == 1
    assert fake_manager.edges[0]["source"] == h1
    assert fake_manager.edges[0]["relation"] == "overlap_score"


def test_neo4j_graph_store_node_exists_checks_manager(tmp_path: Path) -> None:
    fake_manager = _FakeNeo4jManager()
    store = Neo4jGraphStore(manager=fake_manager)

    h = hash_content("neo4j exists check")
    assert store.node_exists(h) is False

    store.upsert_node(h, "document", {})
    assert store.node_exists(h) is True
