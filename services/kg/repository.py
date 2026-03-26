from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class KgJob:
    kg_id: str
    status: str  # queued | running | completed | failed
    created_at: str
    updated_at: str
    doc_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KgRepository:
    """SQLite-backed repository for KG build jobs."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS kg_jobs (
        kg_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        doc_count INTEGER NOT NULL DEFAULT 0,
        node_count INTEGER NOT NULL DEFAULT 0,
        edge_count INTEGER NOT NULL DEFAULT 0,
        artifacts TEXT NOT NULL DEFAULT '{}',
        error_message TEXT
    )
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self._DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_job(self, row: sqlite3.Row) -> KgJob:
        return KgJob(
            kg_id=row["kg_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            doc_count=row["doc_count"],
            node_count=row["node_count"],
            edge_count=row["edge_count"],
            artifacts=json.loads(row["artifacts"]),
            error_message=row["error_message"],
        )

    def create(self, doc_count: int) -> KgJob:
        kg_id = str(uuid.uuid4())
        now = _now()
        job = KgJob(
            kg_id=kg_id,
            status="queued",
            created_at=now,
            updated_at=now,
            doc_count=doc_count,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO kg_jobs
                   (kg_id, status, created_at, updated_at, doc_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (job.kg_id, job.status, job.created_at, job.updated_at, job.doc_count),
            )
        return job

    def get(self, kg_id: str) -> Optional[KgJob]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM kg_jobs WHERE kg_id = ?", (kg_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_all(self) -> List[KgJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_status(self, kg_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE kg_jobs SET status = ?, updated_at = ? WHERE kg_id = ?",
                (status, _now(), kg_id),
            )

    def update_completed(
        self,
        kg_id: str,
        node_count: int,
        edge_count: int,
        artifacts: Dict[str, str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE kg_jobs
                   SET status = 'completed', updated_at = ?,
                       node_count = ?, edge_count = ?, artifacts = ?
                   WHERE kg_id = ?""",
                (_now(), node_count, edge_count, json.dumps(artifacts), kg_id),
            )

    def update_failed(self, kg_id: str, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE kg_jobs
                   SET status = 'failed', updated_at = ?, error_message = ?
                   WHERE kg_id = ?""",
                (_now(), error_message, kg_id),
            )
