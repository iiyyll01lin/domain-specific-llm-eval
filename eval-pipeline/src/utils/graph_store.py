"""Content-hash-addressed graph store with SQLite backend.

This module provides a first-class persistent graph store for the evaluation
pipeline's knowledge graph, replacing the flat JSON-snapshot approach with an
incremental, deduplication-aware store.

Design goals
------------
* **Content-hash addressing** — each node's primary key is the SHA-256 digest
  of its trimmed text content, so identical chunks are never stored twice.
* **Zero new dependencies** — the default backend uses ``sqlite3`` from the
  standard library.
* **Protocol-based interface** — callers depend on the ``GraphStore`` protocol;
  swapping ``SQLiteGraphStore`` for ``Neo4jGraphStore`` (or any future backend)
  requires no changes at call sites.
* **Backward compatibility** — no existing public APIs are removed; this module
  only *adds* functionality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Protocol, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


class NodeRecord(TypedDict):
    """A row from the ``nodes`` table, with properties decoded from JSON."""

    node_hash: str
    node_type: str
    properties: Dict[str, Any]
    created_at: str
    updated_at: str


class RelRecord(TypedDict):
    """A row from the ``relationships`` table, with properties decoded from JSON."""

    src_hash: str
    tgt_hash: str
    rel_type: str
    properties: Dict[str, Any]
    created_at: str


# ---------------------------------------------------------------------------
# Hashing helper
# ---------------------------------------------------------------------------


def hash_content(text: str) -> str:
    """Return a 32-character lowercase hex SHA-256 digest of *text*.

    Leading and trailing whitespace is stripped before hashing so that
    cosmetic differences in chunking do not produce distinct nodes.

    Parameters
    ----------
    text:
        Raw document chunk or any string to be addressed by content.

    Returns
    -------
    str
        A 32-character hex string (first 128 bits of SHA-256).
    """
    normalised = text.strip().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class GraphStore(Protocol):
    """Minimal interface that every graph-store backend must implement."""

    def upsert_node(
        self,
        node_hash: str,
        node_type: str,
        properties: Dict[str, Any],
    ) -> bool:
        """Insert or update a node identified by *node_hash*.

        Returns
        -------
        bool
            ``True`` if the node was **newly created**, ``False`` if it already
            existed (and its properties were updated).
        """
        ...

    def add_relationship(
        self,
        src_hash: str,
        tgt_hash: str,
        rel_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """Add a directed relationship between two nodes.

        The triple ``(src_hash, tgt_hash, rel_type)`` is unique — calling this
        method multiple times with the same triple is idempotent.
        """
        ...

    def node_exists(self, node_hash: str) -> bool:
        """Return ``True`` if a node with *node_hash* is already stored."""
        ...

    def get_all_nodes(self) -> List[NodeRecord]:
        """Return every node currently in the store."""
        ...

    def get_all_relationships(self) -> List[RelRecord]:
        """Return every relationship currently in the store."""
        ...

    def filter_new_hashes(self, candidate_hashes: List[str]) -> List[str]:
        """Return the subset of *candidate_hashes* not yet in the store."""
        ...

    def prune_stale(self, max_age_days: int) -> int:
        """Delete nodes (and their relationships) not updated within *max_age_days*.

        Returns
        -------
        int
            Number of nodes removed.
        """
        ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_hash  TEXT PRIMARY KEY,
    node_type  TEXT NOT NULL,
    properties TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_hash   TEXT NOT NULL REFERENCES nodes(node_hash) ON DELETE CASCADE,
    tgt_hash   TEXT NOT NULL REFERENCES nodes(node_hash) ON DELETE CASCADE,
    rel_type   TEXT NOT NULL,
    properties TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(src_hash, tgt_hash, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(src_hash);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relationships(tgt_hash);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteGraphStore:
    """Content-hash-addressed graph store backed by a local SQLite database.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.  The parent directory
        is created automatically if it does not exist.

    Examples
    --------
    >>> store = SQLiteGraphStore("/tmp/mykg.db")
    >>> h = hash_content("Document chunk text")
    >>> store.upsert_node(h, "document", {"title": "My Doc", "content": "..."})
    True
    >>> store.node_exists(h)
    True
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # GraphStore protocol implementation
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_hash: str,
        node_type: str,
        properties: Dict[str, Any],
    ) -> bool:
        """Insert or update a node.  Auto-generates ``_node_uuid`` when absent.

        Parameters
        ----------
        node_hash:
            SHA-256–derived content-hash (see :func:`hash_content`).
        node_type:
            Semantic type tag, e.g. ``"document"`` or ``"entity"``.
        properties:
            Arbitrary JSON-serialisable dict.  If ``"_node_uuid"`` is absent,
            a stable UUID is synthesised and stored so RAGAS ``Node`` IDs
            remain consistent across runs.

        Returns
        -------
        bool
            ``True`` if the row was inserted (new node), ``False`` if updated.
        """
        props = dict(properties)
        if "_node_uuid" not in props:
            props["_node_uuid"] = str(uuid.uuid4())

        now = _now_iso()
        props_json = json.dumps(props, ensure_ascii=False, default=str)

        with self._conn() as conn:
            existing = conn.execute(
                "SELECT node_hash FROM nodes WHERE node_hash = ?", (node_hash,)
            ).fetchone()

            if existing is None:
                conn.execute(
                    "INSERT INTO nodes (node_hash, node_type, properties, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (node_hash, node_type, props_json, now, now),
                )
                logger.debug("🆕 New node %s (%s)", node_hash[:8], node_type)
                return True
            else:
                conn.execute(
                    "UPDATE nodes SET node_type = ?, properties = ?, updated_at = ? "
                    "WHERE node_hash = ?",
                    (node_type, props_json, now, node_hash),
                )
                logger.debug("♻️  Updated node %s", node_hash[:8])
                return False

    def add_relationship(
        self,
        src_hash: str,
        tgt_hash: str,
        rel_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """Idempotently add a directed relationship.

        If the ``(src_hash, tgt_hash, rel_type)`` triple already exists, the
        call is silently ignored (no update, no error).

        Parameters
        ----------
        src_hash:
            Content-hash of the source node.
        tgt_hash:
            Content-hash of the target node.
        rel_type:
            Relationship label, e.g. ``"jaccard_similarity"``.
        properties:
            Arbitrary JSON-serialisable metadata (scores, shared entities, …).
        """
        now = _now_iso()
        props_json = json.dumps(properties, ensure_ascii=False, default=str)

        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO relationships "
                "(src_hash, tgt_hash, rel_type, properties, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (src_hash, tgt_hash, rel_type, props_json, now),
            )

    def node_exists(self, node_hash: str) -> bool:
        """Return ``True`` if a node with this hash is in the store."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM nodes WHERE node_hash = ? LIMIT 1", (node_hash,)
            ).fetchone()
        return row is not None

    def get_all_nodes(self) -> List[NodeRecord]:
        """Return every node as a :class:`NodeRecord`."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT node_hash, node_type, properties, created_at, updated_at FROM nodes"
            ).fetchall()
        return [
            NodeRecord(
                node_hash=row["node_hash"],
                node_type=row["node_type"],
                properties=json.loads(row["properties"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_all_relationships(self) -> List[RelRecord]:
        """Return every relationship as a :class:`RelRecord`."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT src_hash, tgt_hash, rel_type, properties, created_at "
                "FROM relationships"
            ).fetchall()
        return [
            RelRecord(
                src_hash=row["src_hash"],
                tgt_hash=row["tgt_hash"],
                rel_type=row["rel_type"],
                properties=json.loads(row["properties"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def filter_new_hashes(self, candidate_hashes: List[str]) -> List[str]:
        """Return the subset of *candidate_hashes* not yet present in the store.

        This is the key primitive for incremental builds: callers compute hashes
        for all chunks, then only process the returned subset.

        Parameters
        ----------
        candidate_hashes:
            Hashes to check, in any order.

        Returns
        -------
        List[str]
            Ordered list of hashes from *candidate_hashes* that are new,
            preserving the original insertion order.
        """
        if not candidate_hashes:
            return []

        placeholders = ",".join("?" * len(candidate_hashes))
        with self._conn() as conn:
            existing_rows = conn.execute(
                f"SELECT node_hash FROM nodes WHERE node_hash IN ({placeholders})",
                candidate_hashes,
            ).fetchall()
        existing = {row["node_hash"] for row in existing_rows}
        return [h for h in candidate_hashes if h not in existing]

    def prune_stale(self, max_age_days: int) -> int:
        """Remove nodes (and their cascading relationships) older than *max_age_days*.

        The ``updated_at`` column is used as the recency marker — an ``upsert_node``
        call always refreshes it.

        Parameters
        ----------
        max_age_days:
            Nodes whose ``updated_at`` is older than this many days are removed.

        Returns
        -------
        int
            Number of node rows deleted.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat()

        with self._conn() as conn:
            # Identify stale node hashes before deletion for logging
            stale = conn.execute(
                "SELECT node_hash FROM nodes WHERE updated_at < ?", (cutoff,)
            ).fetchall()
            count = len(stale)
            if count:
                conn.execute("DELETE FROM nodes WHERE updated_at < ?", (cutoff,))
                logger.info("🗑️  Pruned %d stale nodes older than %d days.", count, max_age_days)
        return count


# ---------------------------------------------------------------------------
# Neo4j backend (delegates to Neo4jGraphManager)
# ---------------------------------------------------------------------------


class Neo4jGraphStore:
    """Graph store backend that delegates to :class:`~src.utils.neo4j_manager.Neo4jGraphManager`.

    This class fulfils the :class:`GraphStore` protocol using the existing
    ``Neo4jGraphManager`` as its persistence layer.  It is intentionally thin
    so that the Neo4j manager's richer Cypher query interface remains accessible
    via the ``.manager`` attribute.

    Parameters
    ----------
    manager:
        A connected (or connectable) ``Neo4jGraphManager`` instance.
        Pass a custom manager for testing (dependency injection).
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        # In-memory index of known hashes for O(1) :meth:`node_exists` lookups
        self._known_hashes: set[str] = set()

    def upsert_node(
        self,
        node_hash: str,
        node_type: str,
        properties: Dict[str, Any],
    ) -> bool:
        props = dict(properties)
        if "_node_uuid" not in props:
            props["_node_uuid"] = str(uuid.uuid4())

        is_new = node_hash not in self._known_hashes
        self.manager.add_node(node_hash, node_type=node_type, **props)
        self._known_hashes.add(node_hash)
        return is_new

    def add_relationship(
        self,
        src_hash: str,
        tgt_hash: str,
        rel_type: str,
        properties: Dict[str, Any],
    ) -> None:
        self.manager.add_relationship(src_hash, tgt_hash, rel_type, **properties)

    def node_exists(self, node_hash: str) -> bool:
        return node_hash in self._known_hashes

    def get_all_nodes(self) -> List[NodeRecord]:
        return [
            NodeRecord(
                node_hash=node_id,
                node_type=str(data.get("node_type", "document")),
                properties={k: v for k, v in data.items() if k not in ("id", "node_type")},
                created_at="",
                updated_at="",
            )
            for node_id, data in self.manager.nodes.items()
        ]

    def get_all_relationships(self) -> List[RelRecord]:
        return [
            RelRecord(
                src_hash=str(edge.get("source", "")),
                tgt_hash=str(edge.get("target", "")),
                rel_type=str(edge.get("relation", "")),
                properties={k: v for k, v in edge.get("properties", {}).items()},
                created_at="",
            )
            for edge in self.manager.edges
        ]

    def filter_new_hashes(self, candidate_hashes: List[str]) -> List[str]:
        return [h for h in candidate_hashes if h not in self._known_hashes]

    def prune_stale(self, max_age_days: int) -> int:
        # Neo4j pruning is a no-op in this implementation; apply TTL via
        # native Neo4j APOC procedures in production deployments instead.
        logger.warning(
            "⚠️  prune_stale() is not supported by Neo4jGraphStore. "
            "Use native Neo4j TTL or APOC procedures instead."
        )
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GraphStore",
    "NodeRecord",
    "RelRecord",
    "SQLiteGraphStore",
    "Neo4jGraphStore",
    "hash_content",
]
