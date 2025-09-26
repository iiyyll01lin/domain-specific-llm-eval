from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional


@dataclass
class ProcessingJob:
    job_id: str
    document_id: str
    profile_hash: str
    status: str
    created_at: str
    updated_at: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ProcessingRepository:
    """SQLite-backed repository tracking processing jobs."""

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
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    profile_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_jobs_document
                ON processing_jobs (document_id)
                """
            )
            conn.commit()

    def create_job(self, *, document_id: str, profile_hash: str) -> ProcessingJob:
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = "queued"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processing_jobs (job_id, document_id, profile_hash, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, document_id, profile_hash, status, now, now),
            )
            conn.commit()
        return ProcessingJob(
            job_id=job_id,
            document_id=document_id,
            profile_hash=profile_hash,
            status=status,
            created_at=now,
            updated_at=now,
        )

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, document_id, profile_hash, status, created_at, updated_at, error_code, error_message
                FROM processing_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(self) -> Iterable[ProcessingJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, document_id, profile_hash, status, created_at, updated_at, error_code, error_message
                FROM processing_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM processing_jobs").fetchone()
        return int(row[0]) if row else 0

    def mark_job_running(self, job_id: str) -> None:
        self._update_status(job_id, "running")

    def mark_job_completed(self, job_id: str) -> None:
        self._update_status(job_id, "completed")

    def mark_job_failed(self, job_id: str, *, error_code: str, error_message: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = ?, updated_at = ?, error_code = ?, error_message = ?
                WHERE job_id = ?
                """,
                ("error", now, error_code, error_message, job_id),
            )
            conn.commit()

    def _update_status(self, job_id: str, status: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = ?, updated_at = ?, error_code = NULL, error_message = NULL
                WHERE job_id = ?
                """,
                (status, now, job_id),
            )
            conn.commit()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ProcessingJob:
        payload = dict(row)
        return ProcessingJob(
            job_id=payload["job_id"],
            document_id=payload["document_id"],
            profile_hash=payload["profile_hash"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


__all__ = ["ProcessingRepository", "ProcessingJob"]
