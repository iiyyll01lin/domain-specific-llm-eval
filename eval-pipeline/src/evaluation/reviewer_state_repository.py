from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .reviewer_state_migrations import (
    CURRENT_REVIEWER_STATE_SCHEMA_VERSION,
    postgres_migration_statements,
    sqlite_migration_statements,
)


class ReviewerStateRepository(Protocol):
    def list_queue(
        self,
        *,
        status: Optional[str] = None,
        include_resolved: bool = True,
    ) -> List[Dict[str, Any]]:
        ...

    def upsert_queue_items(self, items: List[Dict[str, Any]]) -> None:
        ...

    def replace_queue(self, items: List[Dict[str, Any]]) -> None:
        ...

    def list_reviewer_results(self) -> List[Dict[str, Any]]:
        ...

    def ingest_reviewer_results(self, items: List[Dict[str, Any]]) -> int:
        ...

    def health(self) -> Dict[str, Any]:
        ...

    def export_backup(self, backup_path: Path) -> Path:
        ...

    def restore_backup(self, backup_path: Path) -> Dict[str, Any]:
        ...

    def list_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        ...


class SQLiteReviewerStateRepository:
    def __init__(
        self,
        db_path: Path,
        review_queue_snapshot_path: Path,
        reviewer_results_snapshot_path: Path,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_queue_snapshot_path = Path(review_queue_snapshot_path)
        self.reviewer_results_snapshot_path = Path(reviewer_results_snapshot_path)
        self._ensure_schema()
        self._apply_migrations()
        self._import_legacy_snapshots_if_needed()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def health(self) -> Dict[str, Any]:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1")
            return {"status": "ok", "backend": "sqlite", "path": str(self.db_path)}
        except sqlite3.Error as exc:
            return {
                "status": "degraded",
                "backend": "sqlite",
                "path": str(self.db_path),
                "error": str(exc),
            }

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                    review_id TEXT PRIMARY KEY,
                    item_index INTEGER,
                    question TEXT NOT NULL,
                    reason TEXT,
                    priority TEXT,
                    status TEXT,
                    suggested_action TEXT,
                    answer TEXT,
                    confidence REAL,
                    ragas_score REAL,
                    keyword_score REAL,
                    reviewer TEXT,
                    reviewer_notes TEXT,
                    resolution TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    resolved_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviewer_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT,
                    item_index INTEGER,
                    question TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    score REAL NOT NULL,
                    notes TEXT,
                    reviewer TEXT NOT NULL,
                    resolution TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(item_index, question, reviewer)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviewer_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    review_id TEXT,
                    reviewer TEXT,
                    tenant_id TEXT,
                    payload_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviewer_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _apply_migrations(self) -> None:
        with self._connect() as connection:
            applied_versions = {
                int(row[0])
                for row in connection.execute(
                    "SELECT version FROM reviewer_schema_migrations"
                ).fetchall()
            }
            for migration in sqlite_migration_statements():
                version = int(migration["version"])
                if version in applied_versions:
                    continue
                for statement in migration["statements"]:
                    try:
                        connection.execute(statement)
                    except sqlite3.OperationalError as exc:
                        if "duplicate column name" not in str(exc).lower():
                            raise
                connection.execute(
                    "INSERT OR IGNORE INTO reviewer_schema_migrations(version) VALUES (?)",
                    (version,),
                )

    def _import_legacy_snapshots_if_needed(self) -> None:
        with self._connect() as connection:
            queue_count = connection.execute(
                "SELECT COUNT(*) FROM review_queue"
            ).fetchone()[0]
            results_count = connection.execute(
                "SELECT COUNT(*) FROM reviewer_results"
            ).fetchone()[0]

        if queue_count == 0 and self.review_queue_snapshot_path.exists():
            queue_items = [
                json.loads(line)
                for line in self.review_queue_snapshot_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if queue_items:
                self.upsert_queue_items(queue_items)

        if results_count == 0 and self.reviewer_results_snapshot_path.exists():
            reviewer_results = [
                json.loads(line)
                for line in self.reviewer_results_snapshot_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if reviewer_results:
                self.ingest_reviewer_results(reviewer_results)

    def list_queue(
        self,
        *,
        status: Optional[str] = None,
        include_resolved: bool = True,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM review_queue"
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_resolved:
            clauses.append("status != ?")
            params.append("resolved")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, review_id ASC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._queue_row_to_dict(row) for row in rows]

    def upsert_queue_items(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as connection:
            for item in items:
                normalized = self._normalize_queue_item(item)
                connection.execute(
                    """
                    INSERT INTO review_queue (
                        review_id, item_index, question, reason, priority, status,
                        suggested_action, answer, confidence, ragas_score, keyword_score,
                        reviewer, reviewer_notes, resolution, created_at, updated_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        item_index=excluded.item_index,
                        question=excluded.question,
                        reason=excluded.reason,
                        priority=excluded.priority,
                        status=excluded.status,
                        suggested_action=excluded.suggested_action,
                        answer=excluded.answer,
                        confidence=excluded.confidence,
                        ragas_score=excluded.ragas_score,
                        keyword_score=excluded.keyword_score,
                        reviewer=excluded.reviewer,
                        reviewer_notes=excluded.reviewer_notes,
                        resolution=excluded.resolution,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        resolved_at=excluded.resolved_at
                    """,
                    (
                        normalized["review_id"],
                        normalized["index"],
                        normalized["question"],
                        normalized["reason"],
                        normalized["priority"],
                        normalized["status"],
                        normalized["suggested_action"],
                        normalized["answer"],
                        normalized["confidence"],
                        normalized["ragas_score"],
                        normalized["keyword_score"],
                        normalized["reviewer"],
                        normalized["reviewer_notes"],
                        normalized["resolution"],
                        normalized["created_at"],
                        normalized["updated_at"],
                        normalized["resolved_at"],
                    ),
                )
                self._record_audit_event(
                    connection,
                    event_type="queue_upsert",
                    review_id=normalized["review_id"],
                    reviewer=normalized["reviewer"],
                    tenant_id=normalized.get("tenant_id"),
                    payload=normalized,
                )
        self._export_snapshots()

    def replace_queue(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM review_queue")
            self._record_audit_event(
                connection,
                event_type="queue_replace",
                review_id=None,
                reviewer=None,
                tenant_id=None,
                payload={"item_count": len(items)},
            )
        self.upsert_queue_items(items)

    def list_reviewer_results(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviewer_results ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self._reviewer_result_row_to_dict(row) for row in rows]

    def ingest_reviewer_results(self, items: List[Dict[str, Any]]) -> int:
        ingested = 0
        with self._connect() as connection:
            for item in items:
                normalized = self._normalize_reviewer_result(item)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO reviewer_results (
                        review_id, item_index, question, approved, score, notes,
                        reviewer, resolution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized.get("review_id"),
                        normalized["index"],
                        normalized["question"],
                        1 if normalized["approved"] else 0,
                        normalized["score"],
                        normalized["notes"],
                        normalized["reviewer"],
                        normalized["resolution"],
                    ),
                )
                if cursor.rowcount > 0:
                    ingested += 1
                    self._record_audit_event(
                        connection,
                        event_type="review_result_ingest",
                        review_id=normalized.get("review_id"),
                        reviewer=normalized["reviewer"],
                        tenant_id=normalized.get("tenant_id"),
                        payload=normalized,
                    )
        self._export_snapshots()
        return ingested

    def _record_audit_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        review_id: Optional[str],
        reviewer: Optional[str],
        tenant_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO reviewer_audit_log (
                event_type, review_id, reviewer, tenant_id, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                review_id,
                reviewer,
                tenant_id,
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def list_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviewer_audit_log ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "review_id": row["review_id"],
                "reviewer": row["reviewer"],
                "tenant_id": row["tenant_id"],
                "payload": json.loads(row["payload_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def export_backup(self, backup_path: Path) -> Path:
        backup_payload = {
            "schema_version": CURRENT_REVIEWER_STATE_SCHEMA_VERSION,
            "backend": "sqlite",
            "queue": self.list_queue(status=None, include_resolved=True),
            "reviewer_results": self.list_reviewer_results(),
            "audit_log": self.list_audit_events(limit=100000),
        }
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(json.dumps(backup_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return backup_path

    def restore_backup(self, backup_path: Path) -> Dict[str, Any]:
        payload = json.loads(Path(backup_path).read_text(encoding="utf-8"))
        self.replace_queue(list(payload.get("queue", [])))
        self.ingest_reviewer_results(list(payload.get("reviewer_results", [])))
        return {
            "restored": True,
            "queue_items": len(payload.get("queue", [])),
            "reviewer_results": len(payload.get("reviewer_results", [])),
        }

    def _export_snapshots(self) -> None:
        queue_items = self.list_queue(status=None, include_resolved=True)
        reviewer_results = self.list_reviewer_results()
        self.review_queue_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.reviewer_results_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_queue_snapshot_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in queue_items)
            + ("\n" if queue_items else ""),
            encoding="utf-8",
        )
        self.reviewer_results_snapshot_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in reviewer_results)
            + ("\n" if reviewer_results else ""),
            encoding="utf-8",
        )

    def _queue_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "review_id": row["review_id"],
            "index": row["item_index"],
            "question": row["question"],
            "reason": row["reason"] or "",
            "priority": row["priority"] or "medium",
            "status": row["status"] or "pending",
            "suggested_action": row["suggested_action"] or "",
            "answer": row["answer"] or "",
            "confidence": row["confidence"],
            "ragas_score": row["ragas_score"],
            "keyword_score": row["keyword_score"],
            "reviewer": row["reviewer"] or "",
            "reviewer_notes": row["reviewer_notes"] or "",
            "resolution": row["resolution"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "resolved_at": row["resolved_at"],
        }

    def _reviewer_result_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "review_id": row["review_id"],
            "index": row["item_index"],
            "question": row["question"],
            "approved": bool(row["approved"]),
            "score": row["score"],
            "notes": row["notes"] or "",
            "reviewer": row["reviewer"],
            "resolution": row["resolution"] or "resolved",
        }

    def _normalize_queue_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "review_id": str(item.get("review_id", "")),
            "index": item.get("index"),
            "question": str(item.get("question", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "priority": str(item.get("priority", "medium")).strip() or "medium",
            "status": str(item.get("status", "pending")).strip() or "pending",
            "suggested_action": str(item.get("suggested_action", "")).strip(),
            "answer": str(item.get("answer", "")),
            "confidence": item.get("confidence"),
            "ragas_score": item.get("ragas_score"),
            "keyword_score": item.get("keyword_score"),
            "reviewer": str(item.get("reviewer", "")).strip(),
            "tenant_id": str(item.get("tenant_id", "")).strip(),
            "reviewer_notes": str(item.get("reviewer_notes", "")).strip(),
            "resolution": str(item.get("resolution", "")).strip(),
            "created_at": str(item.get("created_at", "")),
            "updated_at": str(item.get("updated_at", "")),
            "resolved_at": item.get("resolved_at"),
        }

    def _normalize_reviewer_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "review_id": item.get("review_id"),
            "index": item.get("index"),
            "question": str(item.get("question", "")).strip(),
            "approved": bool(item.get("approved", item.get("is_correct", False))),
            "score": float(item.get("score", 0.0)),
            "notes": str(item.get("notes", "")).strip(),
            "reviewer": str(item.get("reviewer", "unknown")).strip(),
            "tenant_id": str(item.get("tenant_id", "")).strip(),
            "resolution": str(item.get("resolution", "resolved")).strip() or "resolved",
        }


class PostgresReviewerStateRepository:
    def __init__(
        self,
        dsn: str,
        review_queue_snapshot_path: Path,
        reviewer_results_snapshot_path: Path,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 4,
        ssl_mode: str = "require",
        connect_timeout_seconds: int = 10,
    ) -> None:
        self.dsn = str(dsn).strip()
        if not self.dsn:
            raise ValueError("Postgres reviewer repository requires a DSN")
        self.review_queue_snapshot_path = Path(review_queue_snapshot_path)
        self.reviewer_results_snapshot_path = Path(reviewer_results_snapshot_path)
        self.min_pool_size = max(1, int(min_pool_size))
        self.max_pool_size = max(self.min_pool_size, int(max_pool_size))
        self.ssl_mode = str(ssl_mode).strip() or "require"
        self.connect_timeout_seconds = max(1, int(connect_timeout_seconds))
        self._pool = None
        self._ensure_schema()

    def _normalized_dsn(self) -> str:
        dsn = self.dsn
        if "sslmode=" not in dsn:
            dsn = f"{dsn} sslmode={self.ssl_mode}"
        if "connect_timeout=" not in dsn:
            dsn = f"{dsn} connect_timeout={self.connect_timeout_seconds}"
        return dsn

    def _build_pool(self):
        if self._pool is not None:
            return self._pool
        try:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                conninfo=self._normalized_dsn(),
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                open=True,
            )
        except ImportError:
            self._pool = None
        return self._pool

    def _connect(self):
        import psycopg

        pool = self._build_pool()
        if pool is not None:
            return pool.connection()
        return psycopg.connect(self._normalized_dsn())

    def health(self) -> Dict[str, Any]:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return {"status": "ok", "backend": "postgres", "dsn_configured": True}
        except Exception as exc:
            return {
                "status": "degraded",
                "backend": "postgres",
                "dsn_configured": True,
                "pool_enabled": self._pool is not None,
                "ssl_mode": self.ssl_mode,
                "connect_timeout_seconds": self.connect_timeout_seconds,
                "error": str(exc),
            }

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS review_queue (
                        review_id TEXT PRIMARY KEY,
                        item_index INTEGER,
                        question TEXT NOT NULL,
                        reason TEXT,
                        priority TEXT,
                        status TEXT,
                        suggested_action TEXT,
                        answer TEXT,
                        confidence DOUBLE PRECISION,
                        ragas_score DOUBLE PRECISION,
                        keyword_score DOUBLE PRECISION,
                        reviewer TEXT,
                        reviewer_notes TEXT,
                        resolution TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        resolved_at TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reviewer_results (
                        id SERIAL PRIMARY KEY,
                        review_id TEXT,
                        item_index INTEGER,
                        question TEXT NOT NULL,
                        approved BOOLEAN NOT NULL,
                        score DOUBLE PRECISION NOT NULL,
                        notes TEXT,
                        reviewer TEXT NOT NULL,
                        resolution TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(item_index, question, reviewer)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reviewer_audit_log (
                        id BIGSERIAL PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        review_id TEXT,
                        reviewer TEXT,
                        tenant_id TEXT,
                        payload_json JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reviewer_schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            connection.commit()
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version FROM reviewer_schema_migrations")
                applied_versions = {int(row[0]) for row in cursor.fetchall()}
                for migration in postgres_migration_statements():
                    version = int(migration["version"])
                    if version in applied_versions:
                        continue
                    for statement in migration["statements"]:
                        cursor.execute(statement)
                    cursor.execute(
                        "INSERT INTO reviewer_schema_migrations(version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
                        (version,),
                    )
            connection.commit()

    def list_queue(
        self,
        *,
        status: Optional[str] = None,
        include_resolved: bool = True,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM review_queue"
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        elif not include_resolved:
            clauses.append("status != %s")
            params.append("resolved")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, review_id ASC"
        with self._connect() as connection:
            try:
                from psycopg.rows import dict_row

                cursor_context = connection.cursor(row_factory=dict_row)
            except ImportError:
                cursor_context = connection.cursor()
            with cursor_context as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        return [self._queue_row_to_dict(row) for row in rows]

    def upsert_queue_items(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    for item in items:
                        normalized = self._normalize_queue_item(item)
                        cursor.execute(
                            """
                            INSERT INTO review_queue (
                                review_id, item_index, question, reason, priority, status,
                                suggested_action, answer, confidence, ragas_score, keyword_score,
                                reviewer, reviewer_notes, resolution, created_at, updated_at, resolved_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (review_id) DO UPDATE SET
                                item_index=EXCLUDED.item_index,
                                question=EXCLUDED.question,
                                reason=EXCLUDED.reason,
                                priority=EXCLUDED.priority,
                                status=EXCLUDED.status,
                                suggested_action=EXCLUDED.suggested_action,
                                answer=EXCLUDED.answer,
                                confidence=EXCLUDED.confidence,
                                ragas_score=EXCLUDED.ragas_score,
                                keyword_score=EXCLUDED.keyword_score,
                                reviewer=EXCLUDED.reviewer,
                                reviewer_notes=EXCLUDED.reviewer_notes,
                                resolution=EXCLUDED.resolution,
                                created_at=EXCLUDED.created_at,
                                updated_at=EXCLUDED.updated_at,
                                resolved_at=EXCLUDED.resolved_at
                            """,
                            (
                                normalized["review_id"],
                                normalized["index"],
                                normalized["question"],
                                normalized["reason"],
                                normalized["priority"],
                                normalized["status"],
                                normalized["suggested_action"],
                                normalized["answer"],
                                normalized["confidence"],
                                normalized["ragas_score"],
                                normalized["keyword_score"],
                                normalized["reviewer"],
                                normalized["reviewer_notes"],
                                normalized["resolution"],
                                normalized["created_at"],
                                normalized["updated_at"],
                                normalized["resolved_at"],
                            ),
                        )
                        self._record_audit_event(
                            cursor,
                            event_type="queue_upsert",
                            review_id=normalized["review_id"],
                            reviewer=normalized["reviewer"],
                            tenant_id=normalized.get("tenant_id"),
                            payload=normalized,
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self._export_snapshots()

    def replace_queue(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM review_queue")
                self._record_audit_event(
                    cursor,
                    event_type="queue_replace",
                    review_id=None,
                    reviewer=None,
                    tenant_id=None,
                    payload={"item_count": len(items)},
                )
            connection.commit()
        self.upsert_queue_items(items)

    def list_reviewer_results(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            try:
                from psycopg.rows import dict_row

                cursor_context = connection.cursor(row_factory=dict_row)
            except ImportError:
                cursor_context = connection.cursor()
            with cursor_context as cursor:
                cursor.execute(
                    "SELECT * FROM reviewer_results ORDER BY created_at ASC, id ASC"
                )
                rows = cursor.fetchall()
        return [self._reviewer_result_row_to_dict(row) for row in rows]

    def ingest_reviewer_results(self, items: List[Dict[str, Any]]) -> int:
        ingested = 0
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    for item in items:
                        normalized = self._normalize_reviewer_result(item)
                        cursor.execute(
                            """
                            INSERT INTO reviewer_results (
                                review_id, item_index, question, approved, score, notes, reviewer, resolution
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (item_index, question, reviewer) DO NOTHING
                            """,
                            (
                                normalized.get("review_id"),
                                normalized["index"],
                                normalized["question"],
                                normalized["approved"],
                                normalized["score"],
                                normalized["notes"],
                                normalized["reviewer"],
                                normalized["resolution"],
                            ),
                        )
                        if cursor.rowcount > 0:
                            ingested += 1
                            self._record_audit_event(
                                cursor,
                                event_type="review_result_ingest",
                                review_id=normalized.get("review_id"),
                                reviewer=normalized["reviewer"],
                                tenant_id=normalized.get("tenant_id"),
                                payload=normalized,
                            )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self._export_snapshots()
        return ingested

    def _export_snapshots(self) -> None:
        queue_items = self.list_queue(status=None, include_resolved=True)
        reviewer_results = self.list_reviewer_results()
        self.review_queue_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.reviewer_results_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_queue_snapshot_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in queue_items)
            + ("\n" if queue_items else ""),
            encoding="utf-8",
        )
        self.reviewer_results_snapshot_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in reviewer_results)
            + ("\n" if reviewer_results else ""),
            encoding="utf-8",
        )

    def _record_audit_event(
        self,
        cursor,
        *,
        event_type: str,
        review_id: Optional[str],
        reviewer: Optional[str],
        tenant_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        cursor.execute(
            """
            INSERT INTO reviewer_audit_log (
                event_type, review_id, reviewer, tenant_id, payload_json
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                event_type,
                review_id,
                reviewer,
                tenant_id,
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def list_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            try:
                from psycopg.rows import dict_row

                cursor_context = connection.cursor(row_factory=dict_row)
            except ImportError:
                cursor_context = connection.cursor()
            with cursor_context as cursor:
                cursor.execute(
                    "SELECT * FROM reviewer_audit_log ORDER BY id DESC LIMIT %s",
                    (max(1, int(limit)),),
                )
                rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "review_id": row.get("review_id"),
                "reviewer": row.get("reviewer"),
                "tenant_id": row.get("tenant_id"),
                "payload": row.get("payload_json") or {},
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]

    def export_backup(self, backup_path: Path) -> Path:
        backup_payload = {
            "schema_version": CURRENT_REVIEWER_STATE_SCHEMA_VERSION,
            "backend": "postgres",
            "queue": self.list_queue(status=None, include_resolved=True),
            "reviewer_results": self.list_reviewer_results(),
            "audit_log": self.list_audit_events(limit=100000),
        }
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(json.dumps(backup_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return backup_path

    def restore_backup(self, backup_path: Path) -> Dict[str, Any]:
        payload = json.loads(Path(backup_path).read_text(encoding="utf-8"))
        self.replace_queue(list(payload.get("queue", [])))
        self.ingest_reviewer_results(list(payload.get("reviewer_results", [])))
        return {
            "restored": True,
            "queue_items": len(payload.get("queue", [])),
            "reviewer_results": len(payload.get("reviewer_results", [])),
        }

    def _queue_row_to_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(row, dict):
            row = dict(row) if hasattr(row, "keys") else {}
        return {
            "review_id": row["review_id"],
            "index": row["item_index"],
            "question": row["question"],
            "reason": row.get("reason") or "",
            "priority": row.get("priority") or "medium",
            "status": row.get("status") or "pending",
            "suggested_action": row.get("suggested_action") or "",
            "answer": row.get("answer") or "",
            "confidence": row.get("confidence"),
            "ragas_score": row.get("ragas_score"),
            "keyword_score": row.get("keyword_score"),
            "reviewer": row.get("reviewer") or "",
            "tenant_id": row.get("tenant_id") or "",
            "reviewer_notes": row.get("reviewer_notes") or "",
            "resolution": row.get("resolution") or "",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "resolved_at": row.get("resolved_at"),
        }

    def _reviewer_result_row_to_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(row, dict):
            row = dict(row) if hasattr(row, "keys") else {}
        return {
            "review_id": row.get("review_id"),
            "index": row["item_index"],
            "question": row["question"],
            "approved": bool(row["approved"]),
            "score": row["score"],
            "notes": row.get("notes") or "",
            "reviewer": row["reviewer"],
            "tenant_id": row.get("tenant_id") or "",
            "resolution": row.get("resolution") or "resolved",
        }

    def _normalize_queue_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "review_id": str(item.get("review_id", "")),
            "index": item.get("index"),
            "question": str(item.get("question", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "priority": str(item.get("priority", "medium")).strip() or "medium",
            "status": str(item.get("status", "pending")).strip() or "pending",
            "suggested_action": str(item.get("suggested_action", "")).strip(),
            "answer": str(item.get("answer", "")),
            "confidence": item.get("confidence"),
            "ragas_score": item.get("ragas_score"),
            "keyword_score": item.get("keyword_score"),
            "reviewer": str(item.get("reviewer", "")).strip(),
            "tenant_id": str(item.get("tenant_id", "")).strip(),
            "reviewer_notes": str(item.get("reviewer_notes", "")).strip(),
            "resolution": str(item.get("resolution", "")).strip(),
            "created_at": str(item.get("created_at", "")),
            "updated_at": str(item.get("updated_at", "")),
            "resolved_at": item.get("resolved_at"),
        }

    def _normalize_reviewer_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "review_id": item.get("review_id"),
            "index": item.get("index"),
            "question": str(item.get("question", "")).strip(),
            "approved": bool(item.get("approved", item.get("is_correct", False))),
            "score": float(item.get("score", 0.0)),
            "notes": str(item.get("notes", "")).strip(),
            "reviewer": str(item.get("reviewer", "unknown")).strip(),
            "tenant_id": str(item.get("tenant_id", "")).strip(),
            "resolution": str(item.get("resolution", "resolved")).strip() or "resolved",
        }


def build_reviewer_state_repository(
    *,
    backend: str,
    state_store_path: Path,
    review_queue_snapshot_path: Path,
    reviewer_results_snapshot_path: Path,
    state_store_dsn: Optional[str] = None,
    state_store_options: Optional[Dict[str, Any]] = None,
) -> ReviewerStateRepository:
    normalized_backend = str(backend or "sqlite").strip().lower()
    store_options = state_store_options or {}
    if normalized_backend == "postgres":
        return PostgresReviewerStateRepository(
            dsn=str(state_store_dsn or "").strip(),
            review_queue_snapshot_path=review_queue_snapshot_path,
            reviewer_results_snapshot_path=reviewer_results_snapshot_path,
            min_pool_size=int(store_options.get("min_pool_size", 1) or 1),
            max_pool_size=int(store_options.get("max_pool_size", 4) or 4),
            ssl_mode=str(store_options.get("ssl_mode", "require") or "require"),
            connect_timeout_seconds=int(store_options.get("connect_timeout_seconds", 10) or 10),
        )
    return SQLiteReviewerStateRepository(
        db_path=state_store_path,
        review_queue_snapshot_path=review_queue_snapshot_path,
        reviewer_results_snapshot_path=reviewer_results_snapshot_path,
    )