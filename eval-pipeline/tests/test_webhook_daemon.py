from fastapi.testclient import TestClient

import webhook_daemon


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