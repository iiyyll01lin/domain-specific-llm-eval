from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional


@dataclass
class IngestionJob:
    job_id: str
    km_id: str
    version: str
    status: str
    created_at: str
    updated_at: str


class IngestionRepository:
    """Simple SQLite-backed repository for ingestion job metadata."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    km_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_km_version
                ON ingestion_jobs (km_id, version)
                """
            )
            conn.commit()

    def create_job(self, km_id: str, version: str) -> IngestionJob:
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = "queued"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (job_id, km_id, version, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, km_id, version, status, now, now),
            )
            conn.commit()
        return IngestionJob(job_id=job_id, km_id=km_id, version=version, status=status, created_at=now, updated_at=now)

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id, km_id, version, status, created_at, updated_at FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return IngestionJob(**dict(row))

    def list_jobs(self) -> Iterable[IngestionJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, km_id, version, status, created_at, updated_at FROM ingestion_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [IngestionJob(**dict(row)) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM ingestion_jobs").fetchone()
        return int(row[0]) if row else 0
