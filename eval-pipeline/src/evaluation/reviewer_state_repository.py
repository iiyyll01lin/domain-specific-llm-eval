from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        self._import_legacy_snapshots_if_needed()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

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
        self._export_snapshots()

    def replace_queue(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM review_queue")
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
            "resolution": str(item.get("resolution", "resolved")).strip() or "resolved",
        }