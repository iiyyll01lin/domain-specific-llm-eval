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