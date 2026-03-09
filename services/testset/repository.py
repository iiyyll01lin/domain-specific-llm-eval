from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class TestsetJob:
    job_id: str
    config_hash: str
    method: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    sample_count: int = 0
    persona_count: int = 0
    scenario_count: int = 0
    seed: Optional[int] = None
    artifact_prefix: Optional[str] = None
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class TestsetRepository:
    """SQLite-backed repository for testset generation job metadata."""

    __test__ = False

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
                    updated_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    persona_count INTEGER NOT NULL DEFAULT 0,
                    scenario_count INTEGER NOT NULL DEFAULT 0,
                    seed INTEGER,
                    artifact_prefix TEXT,
                    artifact_paths_json TEXT,
                    metadata_json TEXT,
                    error_code TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_testset_jobs_hash
                ON testset_jobs (config_hash)
                """
            )
            self._ensure_additional_columns(conn)
            conn.commit()

    def _ensure_additional_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(testset_jobs)").fetchall()
        }
        required_columns = {
            "sample_count": "INTEGER NOT NULL DEFAULT 0",
            "persona_count": "INTEGER NOT NULL DEFAULT 0",
            "scenario_count": "INTEGER NOT NULL DEFAULT 0",
            "seed": "INTEGER",
            "artifact_prefix": "TEXT",
            "artifact_paths_json": "TEXT",
            "metadata_json": "TEXT",
            "error_code": "TEXT",
            "error_message": "TEXT",
        }
        for name, ddl in required_columns.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE testset_jobs ADD COLUMN {name} {ddl}"
                )

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
                    INSERT INTO testset_jobs (
                        job_id,
                        config_hash,
                        method,
                        status,
                        config_json,
                        created_at,
                        updated_at,
                        sample_count,
                        persona_count,
                        scenario_count,
                        seed,
                        artifact_prefix,
                        artifact_paths_json,
                        metadata_json,
                        error_code,
                        error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, NULL, NULL, '{}', '{}', NULL, NULL)
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
            sample_count=0,
            persona_count=0,
            scenario_count=0,
            seed=None,
            artifact_prefix=None,
            artifact_paths={},
            metadata={},
            error_code=None,
            error_message=None,
        )

    def get_job(self, job_id: str) -> Optional[TestsetJob]:
        with self._connect() as conn:
            row = conn.execute(
                """
          SELECT job_id, config_hash, method, status, config_json, created_at, updated_at,
              sample_count, persona_count, scenario_count, seed, artifact_prefix,
              artifact_paths_json, metadata_json, error_code, error_message
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
          SELECT job_id, config_hash, method, status, config_json, created_at, updated_at,
              sample_count, persona_count, scenario_count, seed, artifact_prefix,
              artifact_paths_json, metadata_json, error_code, error_message
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
                SELECT job_id, config_hash, method, status, config_json, created_at, updated_at,
                       sample_count, persona_count, scenario_count, seed, artifact_prefix,
                       artifact_paths_json, metadata_json, error_code, error_message
                FROM testset_jobs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def mark_running(self, job_id: str) -> None:
        self._update_status(job_id, "running")

    def _update_status(self, job_id: str, status: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE testset_jobs
                SET status = ?,
                    updated_at = ?,
                    error_code = NULL,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (status, now, job_id),
            )
            conn.commit()

    def mark_completed(
        self,
        job_id: str,
        *,
        sample_count: int,
        persona_count: int,
        scenario_count: int,
        seed: Optional[int],
        artifact_prefix: str,
        artifact_paths: Mapping[str, str],
        metadata: Mapping[str, Any],
    ) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE testset_jobs
                SET status = ?,
                    updated_at = ?,
                    sample_count = ?,
                    persona_count = ?,
                    scenario_count = ?,
                    seed = ?,
                    artifact_prefix = ?,
                    artifact_paths_json = ?,
                    metadata_json = ?,
                    error_code = NULL,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (
                    "completed",
                    now,
                    sample_count,
                    persona_count,
                    scenario_count,
                    seed,
                    artifact_prefix,
                    json.dumps(dict(artifact_paths), ensure_ascii=False, sort_keys=True),
                    json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True),
                    job_id,
                ),
            )
            conn.commit()

    def mark_failed(self, job_id: str, *, error_code: str, error_message: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE testset_jobs
                SET status = ?,
                    updated_at = ?,
                    error_code = ?,
                    error_message = ?
                WHERE job_id = ?
                """,
                ("error", now, error_code, error_message, job_id),
            )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM testset_jobs").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> TestsetJob:
        payload = dict(row)
        config_json = payload.get("config_json") or "{}"
        config = json.loads(config_json)
        artifact_paths_json = payload.get("artifact_paths_json") or "{}"
        metadata_json = payload.get("metadata_json") or "{}"
        return TestsetJob(
            job_id=payload["job_id"],
            config_hash=payload["config_hash"],
            method=payload["method"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            config=config,
            sample_count=int(payload.get("sample_count") or 0),
            persona_count=int(payload.get("persona_count") or 0),
            scenario_count=int(payload.get("scenario_count") or 0),
            seed=payload.get("seed"),
            artifact_prefix=payload.get("artifact_prefix"),
            artifact_paths=json.loads(artifact_paths_json),
            metadata=json.loads(metadata_json),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


__all__ = ["TestsetRepository", "TestsetJob"]
