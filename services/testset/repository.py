from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


@dataclass
class TestsetJob:
    job_id: str
    config_hash: str
    method: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]


class TestsetRepository:
    """SQLite-backed repository for testset generation job metadata."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS testset_jobs (
                    job_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_testset_jobs_hash
                ON testset_jobs (config_hash)
                """
            )
            conn.commit()

    def create_job(self, *, config_hash: str, config: Dict[str, Any]) -> TestsetJob:
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = "queued"
        method = str(config.get("method", "unknown"))
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO testset_jobs (job_id, config_hash, method, status, config_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, config_hash, method, status, config_json, now, now),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            existing = self.get_job_by_hash(config_hash)
            if existing is not None:
                return existing
            raise
        return TestsetJob(
            job_id=job_id,
            config_hash=config_hash,
            method=method,
            status=status,
            created_at=now,
            updated_at=now,
            config=json.loads(config_json),
        )

    def get_job(self, job_id: str) -> Optional[TestsetJob]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, config_hash, method, status, config_json, created_at, updated_at
                FROM testset_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def get_job_by_hash(self, config_hash: str) -> Optional[TestsetJob]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, config_hash, method, status, config_json, created_at, updated_at
                FROM testset_jobs
                WHERE config_hash = ?
                """,
                (config_hash,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(self) -> Iterable[TestsetJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, config_hash, method, status, config_json, created_at, updated_at
                FROM testset_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM testset_jobs").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> TestsetJob:
        payload = dict(row)
        config_json = payload.get("config_json") or "{}"
        config = json.loads(config_json)
        return TestsetJob(
            job_id=payload["job_id"],
            config_hash=payload["config_hash"],
            method=payload["method"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            config=config,
        )


__all__ = ["TestsetRepository", "TestsetJob"]
