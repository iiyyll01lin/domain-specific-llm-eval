from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluation.human_feedback_manager import HumanFeedbackManager
from evaluation.reviewer_service import ReviewerWorkflowService
from ui.reviewer_service_client import ReviewerServiceClient


DEFAULT_REVIEWER_SERVICE_CONFIG: Dict[str, Any] = {
    "allowed_tenants": ["default"],
    "required_roles": ["reviewer"],
    "moderation_policy": {
        "blocked_terms": ["password", "ssn", "credit card"],
        "max_notes_length": 2000,
        "require_reason_for_rejection": True,
    },
    "auth": {
        "api_token": "local-dev-reviewer-token",
        "default_reviewer_id": "local-reviewer",
        "default_tenant_id": "default",
        "default_roles": ["reviewer"],
    },
}


def _load_reviewer_config(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    config_path = resolved_base_dir / "config" / "pipeline_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_manager(base_dir: Optional[Path] = None) -> HumanFeedbackManager:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    loaded_config = _load_reviewer_config(resolved_base_dir)
    review_queue_dir = resolved_base_dir / "outputs" / "human_feedback"
    human_feedback_config = (
        loaded_config.get("evaluation", {}).get("human_feedback", {})
        if isinstance(loaded_config, dict)
        else {}
    )
    merged_config = {
        "enabled": True,
        "review_queue_dir": str(review_queue_dir),
        **human_feedback_config,
        "review_queue_dir": str(review_queue_dir),
    }
    return HumanFeedbackManager({"evaluation": {"human_feedback": merged_config}})


def _build_service(base_dir: Optional[Path] = None) -> ReviewerWorkflowService:
    loaded_config = _load_reviewer_config(base_dir)
    manager = _build_manager(base_dir)
    human_feedback_config = (
        loaded_config.get("evaluation", {}).get("human_feedback", {})
        if isinstance(loaded_config, dict)
        else {}
    )
    service_config = {
        **DEFAULT_REVIEWER_SERVICE_CONFIG,
        **human_feedback_config.get("service_boundary", {}),
    }
    return ReviewerWorkflowService(manager, service_config)


def _build_client(base_dir: Optional[Path] = None) -> Optional[ReviewerServiceClient]:
    loaded_config = _load_reviewer_config(base_dir)
    human_feedback_config = (
        loaded_config.get("evaluation", {}).get("human_feedback", {})
        if isinstance(loaded_config, dict)
        else {}
    )
    service_config = {
        **DEFAULT_REVIEWER_SERVICE_CONFIG,
        **human_feedback_config.get("service_boundary", {}),
    }
    service_url = os.environ.get("REVIEWER_SERVICE_URL") or service_config.get("service_url")
    if not service_url:
        return None
    auth_config = service_config.get("auth", {})
    return ReviewerServiceClient(
        service_url=str(service_url),
        api_token=str(os.environ.get("REVIEWER_SERVICE_TOKEN") or auth_config.get("api_token", "")),
        reviewer_id=str(os.environ.get("REVIEWER_ID") or auth_config.get("default_reviewer_id", "reviewer")),
        tenant_id=str(os.environ.get("REVIEWER_TENANT") or auth_config.get("default_tenant_id", "default")),
    )


def _build_auth_context(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    loaded_config = _load_reviewer_config(base_dir)
    human_feedback_config = (
        loaded_config.get("evaluation", {}).get("human_feedback", {})
        if isinstance(loaded_config, dict)
        else {}
    )
    service_config = {
        **DEFAULT_REVIEWER_SERVICE_CONFIG,
        **human_feedback_config.get("service_boundary", {}),
    }
    auth_config = service_config.get("auth", {})
    return {
        "token": str(os.environ.get("REVIEWER_SERVICE_TOKEN") or auth_config.get("api_token", "")),
        "reviewer_id": str(os.environ.get("REVIEWER_ID") or auth_config.get("default_reviewer_id", "reviewer")),
        "tenant_id": str(os.environ.get("REVIEWER_TENANT") or auth_config.get("default_tenant_id", "default")),
        "roles": list(auth_config.get("default_roles", ["reviewer"])),
    }


def list_pending_reviews(
    base_dir: Optional[Path] = None,
    *,
    status: Optional[str] = "pending",
    include_resolved: bool = False,
) -> List[Dict[str, Any]]:
    client = _build_client(base_dir)
    if client is not None:
        payload = client.list_reviews(status=status, include_resolved=include_resolved)
        return payload.get("reviews", [])

    service = _build_service(base_dir)
    auth = service.authenticate(**_build_auth_context(base_dir))
    payload = service.list_reviews(
        auth,
        status=status,
        include_resolved=include_resolved,
    )
    return payload.get("reviews", [])


def submit_reviewer_feedback(
    review_payload: Dict[str, Any],
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    client = _build_client(base_dir)
    if client is not None:
        return client.submit_review(review_payload)

    service = _build_service(base_dir)
    auth = service.authenticate(**_build_auth_context(base_dir))
    return service.submit_review(auth, review_payload)


def get_reviewer_summary(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    client = _build_client(base_dir)
    if client is not None:
        return client.get_summary()

    service = _build_service(base_dir)
    auth = service.authenticate(**_build_auth_context(base_dir))
    return service.get_summary(auth)