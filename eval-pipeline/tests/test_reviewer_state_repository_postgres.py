from __future__ import annotations

from pathlib import Path

from src.evaluation.reviewer_state_repository import (
    PostgresReviewerStateRepository,
    build_reviewer_state_repository,
)


class _FakeCursor:
    def __init__(self) -> None:
        self.rowcount = 1

    def execute(self, _query, _params=None):
        return None

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def cursor(self, **_kwargs):
        return _FakeCursor()

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