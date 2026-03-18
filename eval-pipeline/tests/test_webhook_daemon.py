from fastapi.testclient import TestClient

import webhook_daemon
from src.evaluation.human_feedback_manager import HumanFeedbackManager


AUTH_HEADERS = {
    "X-Reviewer-Token": "local-dev-reviewer-token",
    "X-Reviewer-Id": "api-reviewer",
    "X-Tenant-Id": "default",
}


def test_webhook_queues_supported_push(monkeypatch):
    captured = []

    def fake_run(payload):
        captured.append(payload.model_dump())
        class _Result:
            returncode = 0
            stderr = ""
        return _Result()

    monkeypatch.setattr(webhook_daemon, "run_evaluation_pipeline", fake_run)
    client = TestClient(webhook_daemon.app)

    response = client.post(
        "/webhook",
        json={"event_type": "push", "ref": "refs/heads/main", "docs": 4, "samples": 8},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert captured[0]["docs"] == 4


def test_webhook_ignores_unsupported_ref():
    client = TestClient(webhook_daemon.app)

    response = client.post(
        "/webhook",
        json={"event_type": "push", "ref": "refs/tags/v1.0.0"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_reviewer_api_lists_and_resolves_pending_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(webhook_daemon, "eval_pipeline_dir", tmp_path)

    manager = HumanFeedbackManager(
        {
            "evaluation": {
                "human_feedback": {
                    "enabled": True,
                    "threshold": 0.7,
                    "review_queue_dir": str(tmp_path / "outputs" / "human_feedback"),
                }
            }
        }
    )
    manager.process_feedback(
        {"questions": ["Question 1"]},
        [
            {
                "answer": "Short answer",
                "confidence": 0.2,
                "ragas_score": 0.1,
                "keyword_score": 0.8,
                "domain_score": 0.1,
            }
        ],
    )

    client = TestClient(webhook_daemon.app)
    list_response = client.get("/reviews", headers=AUTH_HEADERS)

    assert list_response.status_code == 200
    assert list_response.json()["summary"]["pending_reviews"] == 1

    queued_review = list_response.json()["reviews"][0]
    submit_response = client.post(
        "/reviews/submit",
        json={
            "review_id": queued_review["review_id"],
            "approved": True,
            "score": 1.0,
            "notes": "Approved through API",
            "reviewer": "api-reviewer",
        },
        headers=AUTH_HEADERS,
    )

    assert submit_response.status_code == 200
    assert submit_response.json()["submitted"] is True

    summary_response = client.get("/reviews/summary", headers=AUTH_HEADERS)
    assert summary_response.status_code == 200
    assert summary_response.json()["pending_reviews"] == 0
    assert summary_response.json()["resolved_reviews"] == 1


def test_reviewer_api_rejects_missing_auth_headers() -> None:
    client = TestClient(webhook_daemon.app)

    response = client.get("/reviews")

    assert response.status_code == 401