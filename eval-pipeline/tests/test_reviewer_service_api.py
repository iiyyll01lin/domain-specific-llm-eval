from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.evaluation.human_feedback_manager import HumanFeedbackManager
from src.services.reviewer_api import create_reviewer_service_app


AUTH_HEADERS = {
    "X-Reviewer-Token": "local-dev-reviewer-token",
    "X-Reviewer-Id": "api-reviewer",
    "X-Tenant-Id": "default",
    "X-Reviewer-Roles": "reviewer",
}


def test_reviewer_service_api_health_and_queue_flow(tmp_path: Path) -> None:
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
        [{"answer": "Short answer", "confidence": 0.2, "ragas_score": 0.1, "keyword_score": 0.8, "domain_score": 0.1}],
    )

    app = create_reviewer_service_app(tmp_path)
    client = TestClient(app)

    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"
    list_response = client.get("/reviews", headers=AUTH_HEADERS)
    assert list_response.status_code == 200
    queued_review = list_response.json()["reviews"][0]

    submit_response = client.post(
        "/reviews/submit",
        headers=AUTH_HEADERS,
        json={
            "review_id": queued_review["review_id"],
            "approved": True,
            "score": 1.0,
            "notes": "Approved through standalone service",
        },
    )

    assert submit_response.status_code == 200
    assert submit_response.json()["submitted"] is True


def test_reviewer_service_contract_file_exists() -> None:
    contract_path = Path("/data/yy/domain-specific-llm-eval/eval-pipeline/contracts/reviewer_service_contract.yaml")
    content = contract_path.read_text(encoding="utf-8")

    assert contract_path.exists()
    assert "X-Reviewer-Token" in content
    assert "moderationDecisionSchema" in content
    assert "persistenceSchema" in content
    assert "/readyz" in content


def test_reviewer_service_api_rate_limits_when_configured(tmp_path: Path, monkeypatch) -> None:
    principal_file = tmp_path / "reviewer_principals.json"
    principal_file.write_text(
        '{"tokens": {"rate-token": {"reviewer_id": "rate-reviewer", "tenant_ids": ["default"], "roles": ["reviewer"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("REVIEWER_AUTH_SOURCE_TYPE", "principal-file")
    monkeypatch.setenv("REVIEWER_PRINCIPAL_FILE", str(principal_file))

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline_config.yaml").write_text(
        "evaluation:\n  human_feedback:\n    service_boundary:\n      rate_limit_rpm: 1\n",
        encoding="utf-8",
    )

    app = create_reviewer_service_app(tmp_path)
    client = TestClient(app)
    headers = {
        "X-Reviewer-Token": "rate-token",
        "X-Reviewer-Id": "rate-reviewer",
        "X-Tenant-Id": "default",
    }

    assert client.get("/reviews", headers=headers).status_code == 200
    assert client.get("/reviews", headers=headers).status_code == 429