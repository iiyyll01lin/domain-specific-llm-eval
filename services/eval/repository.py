from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from services.eval.run_states import RunState


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    testset_id: str
    profile: str
    status: str
    created_at: str
    updated_at: str
    rag_endpoint: Optional[str]
    timeout_seconds: int
    max_retries: int


class EvaluationRunRepository:
    """SQLite-backed repository for evaluation run metadata."""

    _ACTIVE_STATES = (RunState.PENDING.value, RunState.RUNNING.value)

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
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    testset_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rag_endpoint TEXT,
                    timeout_seconds INTEGER NOT NULL,
                    max_retries INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_testset_profile
                ON evaluation_runs (testset_id, profile)
                """
            )
            conn.commit()

    def create_run(
        self,
        *,
        testset_id: str,
        profile: str,
        rag_endpoint: Optional[str],
        timeout_seconds: int,
        max_retries: int,
    ) -> EvaluationRun:
        run_id = uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        status = RunState.PENDING.value
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id,
                    testset_id,
                    profile,
                    status,
                    rag_endpoint,
                    timeout_seconds,
                    max_retries,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    testset_id,
                    profile,
                    status,
                    rag_endpoint,
                    int(timeout_seconds),
                    int(max_retries),
                    now,
                    now,
                ),
            )
            conn.commit()
        return EvaluationRun(
            run_id=run_id,
            testset_id=testset_id,
            profile=profile,
            status=status,
            created_at=now,
            updated_at=now,
            rag_endpoint=rag_endpoint,
            timeout_seconds=int(timeout_seconds),
            max_retries=int(max_retries),
        )

    def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, testset_id, profile, status, rag_endpoint, timeout_seconds, max_retries, created_at, updated_at
                FROM evaluation_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def get_active_run(self, *, testset_id: str, profile: str) -> Optional[EvaluationRun]:
        active_states = self._ACTIVE_STATES
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, testset_id, profile, status, rag_endpoint, timeout_seconds, max_retries, created_at, updated_at
                FROM evaluation_runs
                WHERE testset_id = ?
                  AND profile = ?
                  AND status IN (?, ?)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (testset_id, profile, *active_states),
            ).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def list_runs(self) -> Iterable[EvaluationRun]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, testset_id, profile, status, rag_endpoint, timeout_seconds, max_retries, created_at, updated_at
                FROM evaluation_runs
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def update_status(self, run_id: str, *, status: str) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE evaluation_runs
                SET status = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (status, now, run_id),
            )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM evaluation_runs").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> EvaluationRun:
        payload = dict(row)
        return EvaluationRun(
            run_id=payload["run_id"],
            testset_id=payload["testset_id"],
            profile=payload["profile"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            rag_endpoint=payload.get("rag_endpoint"),
            timeout_seconds=int(payload["timeout_seconds"]),
            max_retries=int(payload["max_retries"]),
        )


__all__ = ["EvaluationRun", "EvaluationRunRepository"]
