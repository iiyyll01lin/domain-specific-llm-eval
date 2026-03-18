from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .human_feedback_manager import HumanFeedbackManager
from .reviewer_auth import build_reviewer_auth_source


@dataclass(frozen=True)
class ReviewerAuthContext:
    token: str
    reviewer_id: str
    tenant_id: str
    roles: List[str]


class ReviewerWorkflowService:
    def __init__(self, manager: HumanFeedbackManager, config: Dict[str, Any]):
        self.manager = manager
        self.config = config
        self.moderation_policy = config.get("moderation_policy", {})
        self.allowed_tenants = set(config.get("allowed_tenants", ["default"]))
        self.required_roles = set(config.get("required_roles", ["reviewer"]))
        self.auth_source = build_reviewer_auth_source(config)

    def authenticate(
        self,
        *,
        token: str,
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[List[str]] = None,
    ) -> ReviewerAuthContext:
        principal = self.auth_source.authenticate(
            token=token,
            reviewer_id=reviewer_id,
            tenant_id=tenant_id,
            roles=roles,
        )
        resolved_tenant = str(tenant_id or (principal.tenant_ids[0] if principal.tenant_ids else "default")).strip()
        if resolved_tenant not in self.allowed_tenants:
            raise PermissionError(f"Tenant '{resolved_tenant}' is not allowed")
        if principal.tenant_ids and resolved_tenant not in principal.tenant_ids:
            raise PermissionError(
                f"Reviewer '{principal.reviewer_id}' is not a member of tenant '{resolved_tenant}'"
            )
        if self.required_roles and not (self.required_roles & set(principal.roles)):
            raise PermissionError("Reviewer role requirements not satisfied")

        return ReviewerAuthContext(
            token=token,
            reviewer_id=principal.reviewer_id,
            tenant_id=resolved_tenant,
            roles=principal.roles,
        )

    def health(self) -> Dict[str, Any]:
        auth_health = self.auth_source.health()
        repository = getattr(self.manager, "repository", None)
        repository_health = (
            repository.health() if repository and hasattr(repository, "health") else {"status": "unknown"}
        )
        overall_status = "ok"
        if auth_health.get("status") not in {"ok", "ready"} or repository_health.get("status") not in {"ok", "ready"}:
            overall_status = "degraded"
        return {
            "status": overall_status,
            "auth": auth_health,
            "repository": repository_health,
        }

    def list_reviews(
        self,
        auth: ReviewerAuthContext,
        *,
        status: Optional[str],
        include_resolved: bool,
    ) -> Dict[str, Any]:
        return {
            "tenant_id": auth.tenant_id,
            "reviewer_id": auth.reviewer_id,
            "reviews": self.manager.list_review_queue(
                status=status,
                include_resolved=include_resolved,
            ),
            "summary": self.manager.get_review_summary(),
        }

    def get_summary(self, auth: ReviewerAuthContext) -> Dict[str, Any]:
        summary = self.manager.get_review_summary()
        summary.update(
            {
                "tenant_id": auth.tenant_id,
                "reviewer_id": auth.reviewer_id,
            }
        )
        return summary

    def submit_review(
        self,
        auth: ReviewerAuthContext,
        review_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        moderation_result = self._moderate_submission(review_payload)
        if not moderation_result["allowed"]:
            return {
                "submitted": False,
                "moderation": moderation_result,
                "summary": self.manager.get_review_summary(),
            }

        normalized_payload = dict(review_payload)
        normalized_payload["reviewer"] = auth.reviewer_id
        normalized_payload["tenant_id"] = auth.tenant_id
        result = self.manager.submit_reviewer_decision(normalized_payload)
        result["tenant_id"] = auth.tenant_id
        result["reviewer_id"] = auth.reviewer_id
        result["moderation"] = moderation_result
        return result

    def _moderate_submission(self, review_payload: Dict[str, Any]) -> Dict[str, Any]:
        notes = str(review_payload.get("notes", ""))
        blocked_terms = [str(term).lower() for term in self.moderation_policy.get("blocked_terms", [])]
        max_notes_length = self._safe_int(
            self.moderation_policy.get("max_notes_length", 2000),
            default=2000,
        )
        require_reason_for_rejection = bool(
            self.moderation_policy.get("require_reason_for_rejection", True)
        )

        violations: List[str] = []
        if len(notes) > max_notes_length:
            violations.append("notes_too_long")
        lower_notes = notes.lower()
        matched_terms = [term for term in blocked_terms if term and term in lower_notes]
        if matched_terms:
            violations.append("blocked_terms_detected")
        approved = bool(review_payload.get("approved", False))
        if not approved and require_reason_for_rejection and not notes.strip():
            violations.append("rejection_requires_notes")

        return {
            "allowed": not violations,
            "violations": violations,
            "matched_terms": matched_terms,
        }

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default