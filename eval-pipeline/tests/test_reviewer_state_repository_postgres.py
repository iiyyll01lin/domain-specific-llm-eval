from __future__ import annotations

from pathlib import Path

from src.evaluation.reviewer_state_repository import (
    PostgresReviewerStateRepository,
    build_reviewer_state_repository,
)


class _FakeCursor:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.rowcount = 1
        self._results = []

    def execute(self, query, _params=None):
        normalized = " ".join(str(query).split()).lower()
        if "select version from reviewer_schema_migrations" in normalized:
            self._results = []
        elif "select * from reviewer_audit_log" in normalized:
            self._results = list(self.connection.audit_rows)
        elif "insert into reviewer_audit_log" in normalized:
            self.connection.audit_rows.append(
                {
                    "id": len(self.connection.audit_rows) + 1,
                    "event_type": "queue_upsert",
                    "review_id": "review-1",
                    "reviewer": "alice",
                    "tenant_id": "tenant-a",
                    "payload_json": {"review_id": "review-1"},
                    "created_at": "2026-03-18T00:00:00Z",
                }
            )
        elif "select * from review_queue" in normalized or "select * from reviewer_results" in normalized:
            self._results = []
        return None

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.audit_rows = []

    def cursor(self, **_kwargs):
        return _FakeCursor(self)

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_postgres_repository_health_and_factory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        PostgresReviewerStateRepository,
        "_connect",
        lambda self: _FakeConnection(),
    )

    repository = build_reviewer_state_repository(
        backend="postgres",
        state_store_path=tmp_path / "ignored.db",
        state_store_dsn="postgresql://reviewer:secret@db/reviewer_service",
        review_queue_snapshot_path=tmp_path / "review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "reviewer_results.jsonl",
    )

    health = repository.health()

    assert health["status"] == "ok"
    assert health["backend"] == "postgres"


def test_postgres_repository_backup_and_audit_contract(monkeypatch, tmp_path: Path) -> None:
    fake_connection = _FakeConnection()
    monkeypatch.setattr(
        PostgresReviewerStateRepository,
        "_connect",
        lambda self: fake_connection,
    )

    repository = PostgresReviewerStateRepository(
        dsn="postgresql://reviewer:secret@db/reviewer_service",
        review_queue_snapshot_path=tmp_path / "review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "reviewer_results.jsonl",
    )
    repository.upsert_queue_items(
        [
            {
                "review_id": "review-1",
                "index": 1,
                "question": "Question 1",
                "reviewer": "alice",
                "tenant_id": "tenant-a",
            }
        ]
    )

    audit_events = repository.list_audit_events(limit=10)
    backup_path = repository.export_backup(tmp_path / "backup.json")

    assert audit_events
    assert audit_events[0]["event_type"] == "queue_upsert"
    assert backup_path.exists()