from __future__ import annotations

from pathlib import Path

from src.evaluation.reviewer_state_repository import SQLiteReviewerStateRepository


def test_sqlite_reviewer_state_repository_round_trips_queue_and_results(tmp_path: Path) -> None:
    repository = SQLiteReviewerStateRepository(
        db_path=tmp_path / "reviewer_state.db",
        review_queue_snapshot_path=tmp_path / "review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "reviewer_results.jsonl",
    )

    repository.upsert_queue_items(
        [
            {
                "review_id": "review-1",
                "index": 0,
                "question": "Question 1",
                "reason": "Low confidence score: 0.2",
                "priority": "high",
                "status": "pending",
                "suggested_action": "Review",
                "answer": "Short answer",
                "created_at": "2026-03-18T10:00:00",
                "updated_at": "2026-03-18T10:00:00",
            }
        ]
    )
    ingested = repository.ingest_reviewer_results(
        [
            {
                "review_id": "review-1",
                "index": 0,
                "question": "Question 1",
                "approved": True,
                "score": 1.0,
                "notes": "Approved",
                "reviewer": "repo-tester",
                "resolution": "resolved",
            }
        ]
    )

    queue = repository.list_queue(status=None, include_resolved=True)
    results = repository.list_reviewer_results()

    assert ingested == 1
    assert queue[0]["review_id"] == "review-1"
    assert results[0]["reviewer"] == "repo-tester"
    assert (tmp_path / "review_queue.jsonl").exists()
    assert (tmp_path / "reviewer_results.jsonl").exists()


def test_sqlite_reviewer_state_repository_restores_backup_and_replays_audit_log(tmp_path: Path) -> None:
    primary = SQLiteReviewerStateRepository(
        db_path=tmp_path / "primary.db",
        review_queue_snapshot_path=tmp_path / "primary_review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "primary_reviewer_results.jsonl",
    )
    primary.replace_queue(
        [
            {
                "review_id": "review-restore-1",
                "index": 1,
                "question": "Question Restore 1",
                "reason": "Low confidence score: 0.4",
                "priority": "high",
                "status": "pending",
                "reviewer": "alice",
                "tenant_id": "tenant-a",
                "created_at": "2026-03-18T12:00:00",
                "updated_at": "2026-03-18T12:00:00",
            }
        ]
    )
    primary.ingest_reviewer_results(
        [
            {
                "review_id": "review-restore-1",
                "index": 1,
                "question": "Question Restore 1",
                "approved": True,
                "score": 0.9,
                "notes": "Recovered approval",
                "reviewer": "alice",
                "tenant_id": "tenant-a",
                "resolution": "resolved",
                "created_at": "2026-03-18T12:01:00",
            }
        ]
    )

    backup_path = primary.export_backup(tmp_path / "reviewer_backup.json")
    backup_payload = __import__("json").loads(backup_path.read_text(encoding="utf-8"))

    restored = SQLiteReviewerStateRepository(
        db_path=tmp_path / "restored.db",
        review_queue_snapshot_path=tmp_path / "restored_review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "restored_reviewer_results.jsonl",
    )
    restore_result = restored.restore_backup(backup_path)

    replayed = SQLiteReviewerStateRepository(
        db_path=tmp_path / "replayed.db",
        review_queue_snapshot_path=tmp_path / "replayed_review_queue.jsonl",
        reviewer_results_snapshot_path=tmp_path / "replayed_reviewer_results.jsonl",
    )
    replay_result = replayed.replay_audit_events(backup_payload["audit_log"], reset_state=True)

    assert restore_result["restored"] is True
    assert restore_result["audit_events"] == len(backup_payload["audit_log"])
    assert restored.list_queue(status=None, include_resolved=True)[0]["tenant_id"] == "tenant-a"
    assert restored.list_reviewer_results()[0]["tenant_id"] == "tenant-a"
    assert len(restored.list_audit_events(limit=100)) == len(backup_payload["audit_log"])

    assert replay_result["replayed"] >= 2
    assert replayed.list_queue(status=None, include_resolved=True)[0]["review_id"] == "review-restore-1"
    assert replayed.list_reviewer_results()[0]["review_id"] == "review-restore-1"