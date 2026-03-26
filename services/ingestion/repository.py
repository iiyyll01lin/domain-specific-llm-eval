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
    document_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    deduplicated: bool = False


@dataclass
class DocumentRecord:
    document_id: str
    km_id: str
    version: str
    checksum: str
    storage_key: str
    size_bytes: int
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
                    updated_at TEXT NOT NULL,
                    document_id TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    deduplicated INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_column(conn, "ingestion_jobs", "document_id", "TEXT")
            self._ensure_column(conn, "ingestion_jobs", "error_code", "TEXT")
            self._ensure_column(conn, "ingestion_jobs", "error_message", "TEXT")
            self._ensure_column(conn, "ingestion_jobs", "deduplicated", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_km_version
                ON ingestion_jobs (km_id, version)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    document_id TEXT PRIMARY KEY,
                    km_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ingested_documents_km_version
                ON ingested_documents (km_id, version)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ingested_documents_checksum
                ON ingested_documents (checksum)
                """
            )
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_job(self, km_id: str, version: str) -> IngestionJob:
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = "queued"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (job_id, km_id, version, status, created_at, updated_at, document_id, error_code, error_message, deduplicated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (job_id, km_id, version, status, now, now, None, None, None),
            )
            conn.commit()
        return IngestionJob(
            job_id=job_id,
            km_id=km_id,
            version=version,
            status=status,
            created_at=now,
            updated_at=now,
        )

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, km_id, version, status, created_at, updated_at, document_id, error_code, error_message, deduplicated
                FROM ingestion_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(self) -> Iterable[IngestionJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, km_id, version, status, created_at, updated_at, document_id, error_code, error_message, deduplicated
                FROM ingestion_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM ingestion_jobs").fetchone()
        return int(row[0]) if row else 0

    def mark_job_running(self, job_id: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                "UPDATE ingestion_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                ("running", now, job_id),
            )
            conn.commit()

    def mark_job_failed(self, job_id: str, error_code: str, error_message: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, updated_at = ?, error_code = ?, error_message = ?, deduplicated = 0
                WHERE job_id = ?
                """,
                ("error", now, error_code, error_message, job_id),
            )
            conn.commit()

    def mark_job_completed(self, job_id: str, document_id: str, *, deduplicated: bool) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = "duplicate" if deduplicated else "completed"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, updated_at = ?, document_id = ?, error_code = NULL, error_message = NULL, deduplicated = ?
                WHERE job_id = ?
                """,
                (status, now, document_id, 1 if deduplicated else 0, job_id),
            )
            conn.commit()

    def create_document(
        self,
        *,
        document_id: str,
        km_id: str,
        version: str,
        checksum: str,
        storage_key: str,
        size_bytes: int,
    ) -> DocumentRecord:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingested_documents (document_id, km_id, version, checksum, storage_key, size_bytes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, km_id, version, checksum, storage_key, size_bytes, now, now),
            )
            conn.commit()
        return DocumentRecord(
            document_id=document_id,
            km_id=km_id,
            version=version,
            checksum=checksum,
            storage_key=storage_key,
            size_bytes=size_bytes,
            created_at=now,
            updated_at=now,
        )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, km_id, version, checksum, storage_key, size_bytes, created_at, updated_at
                FROM ingested_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def get_document_by_km_version(self, km_id: str, version: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, km_id, version, checksum, storage_key, size_bytes, created_at, updated_at
                FROM ingested_documents
                WHERE km_id = ? AND version = ?
                """,
                (km_id, version),
            ).fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def get_document_by_checksum(self, checksum: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT document_id, km_id, version, checksum, storage_key, size_bytes, created_at, updated_at
                FROM ingested_documents
                WHERE checksum = ?
                """,
                (checksum,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def list_documents(self) -> Iterable[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, km_id, version, checksum, storage_key, size_bytes, created_at, updated_at
                FROM ingested_documents
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> IngestionJob:
        payload = dict(row)
        return IngestionJob(
            job_id=payload["job_id"],
            km_id=payload["km_id"],
            version=payload["version"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            document_id=payload.get("document_id"),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
            deduplicated=bool(payload.get("deduplicated", 0)),
        )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        payload = dict(row)
        return DocumentRecord(
            document_id=payload["document_id"],
            km_id=payload["km_id"],
            version=payload["version"],
            checksum=payload["checksum"],
            storage_key=payload["storage_key"],
            size_bytes=int(payload["size_bytes"]),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
        )
