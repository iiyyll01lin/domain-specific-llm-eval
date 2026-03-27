"""ProposalStore — SQLite-backed staging area for graph repair proposals.

Proposals are written here by the Graph Engineer node and remain immutable
until a human approves or rejects them via the insights-portal.  The actual
upsert to ``SQLiteGraphStore`` only happens after approval.

Table schema
------------
``proposals``
    proposal_id  TEXT PK
    thread_id    TEXT NOT NULL     — LangGraph thread_id for checkpoint resume
    state_json   TEXT NOT NULL     — JSON-serialised HealingState at await point
    status       TEXT              — 'pending' | 'approved' | 'rejected' | 'committed' | 'expired'
    created_at   TEXT
    expires_at   TEXT              — proposals auto-expire after TTL_HOURS (default 48)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_TTL_HOURS = 48

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id  TEXT PRIMARY KEY,
    thread_id    TEXT NOT NULL,
    state_json   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_thread ON proposals(thread_id);
"""


class ProposalStore:
    """Persist and retrieve agentic repair proposals.

    Parameters
    ----------
    db_path:
        File path for the proposals SQLite database.  The parent directory is
        created automatically.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_proposal(
        self,
        proposal_id: str,
        thread_id: str,
        state: Dict[str, Any],
        ttl_hours: int = _TTL_HOURS,
    ) -> None:
        """Upsert a proposal into the staging table."""
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO proposals
                   (proposal_id, thread_id, state_json, status, created_at, expires_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (proposal_id, thread_id, json.dumps(state, default=str), now.isoformat(), expires_at),
            )
        logger.info("Staged proposal %s (thread=%s)", proposal_id, thread_id)

    def update_status(self, proposal_id: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE proposals SET status=? WHERE proposal_id=?",
                (status, proposal_id),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_pending(self) -> List[Dict[str, Any]]:
        """Return all non-expired pending proposals, newest first."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM proposals WHERE status='pending' AND expires_at > ?"
                " ORDER BY created_at DESC",
                (now,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)
            ).fetchone()
        return self._decode_row(row) if row else None

    def get_thread_id(self, proposal_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT thread_id FROM proposals WHERE proposal_id=?", (proposal_id,)
            ).fetchone()
        return str(row["thread_id"]) if row else None

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Delete proposals past their TTL.  Returns count of removed rows."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM proposals WHERE expires_at <= ?", (now,)
            )
        pruned = cursor.rowcount
        if pruned:
            logger.info("Pruned %d expired proposals.", pruned)
        return pruned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(row)
        raw = d.pop("state_json", "{}")
        try:
            d["state"] = json.loads(raw)
        except json.JSONDecodeError:
            d["state"] = {}
        return d
