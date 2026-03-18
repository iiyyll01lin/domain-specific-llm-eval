from __future__ import annotations

from pathlib import Path

from src.evaluation.human_feedback_manager import HumanFeedbackManager
from src.evaluation.reviewer_service import ReviewerWorkflowService


def _build_service(tmp_path: Path) -> ReviewerWorkflowService:
    manager = HumanFeedbackManager(
        {
            "evaluation": {
                "human_feedback": {
                    "enabled": True,
                    "threshold": 0.7,
                    "review_queue_dir": str(tmp_path),
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
    return ReviewerWorkflowService(
        manager,
        {
            "allowed_tenants": ["tenant-a"],
            "required_roles": ["reviewer"],
            "auth": {
                "api_token": "token-a",
                "default_reviewer_id": "reviewer-a",
                "default_tenant_id": "tenant-a",
                "default_roles": ["reviewer"],
            },
            "moderation_policy": {
                "blocked_terms": ["password"],
                "max_notes_length": 50,
                "require_reason_for_rejection": True,
            },
        },
    )


def test_reviewer_service_authenticates_and_lists_reviews(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    auth = service.authenticate(
        token="token-a",
        reviewer_id="alice",
        tenant_id="tenant-a",
        roles=["reviewer"],
    )

    payload = service.list_reviews(auth, status="pending", include_resolved=False)

    assert payload["tenant_id"] == "tenant-a"
    assert payload["reviewer_id"] == "alice"
    assert payload["reviews"]


def test_reviewer_service_rejects_wrong_tenant(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    try:
        service.authenticate(
            token="token-a",
            reviewer_id="alice",
            tenant_id="tenant-b",
            roles=["reviewer"],
        )
    except PermissionError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("Expected tenant authentication failure")


def test_reviewer_service_blocks_moderation_violations(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    auth = service.authenticate(
        token="token-a",
        reviewer_id="alice",
        tenant_id="tenant-a",
        roles=["reviewer"],
    )

    result = service.submit_review(
        auth,
        {
            "approved": False,
            "notes": "password leaked",
        },
    )

    assert result["submitted"] is False
    assert result["moderation"]["allowed"] is False
    assert "blocked_terms_detected" in result["moderation"]["violations"]