from __future__ import annotations

import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.evaluation.reviewer_token_issuer import ReviewerTokenIssuerService
from src.ui.reviewer_actions import _build_service
from src.ui.dashboard_data import (
    build_artifact_diff_view,
    build_observability_triage_queue,
    get_issue_cluster_drilldown,
    search_observability_artifacts,
)


logger = logging.getLogger(__name__)


class ReviewerDecisionPayload(BaseModel):
    review_id: Optional[str] = None
    index: Optional[int] = None
    question: str = Field(default="")
    approved: bool = Field(default=True)
    score: Optional[float] = None
    notes: str = Field(default="")
    reviewer: str = Field(default="reviewer")
    resolution: str = Field(default="resolved")


class ReviewerTokenIssuePayload(BaseModel):
    reviewer_id: str
    tenant_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=lambda: ["reviewer"])
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewerTokenRevokePayload(BaseModel):
    token: Optional[str] = None
    jti: Optional[str] = None
    reason: str = Field(default="manual_revocation")


def _build_token_issuer(resolved_base_dir: Path) -> ReviewerTokenIssuerService:
    issuer_dir = resolved_base_dir / "outputs" / "reviewer_service"
    issuer_dir.mkdir(parents=True, exist_ok=True)
    return ReviewerTokenIssuerService(
        issuer=os.environ.get("REVIEWER_TOKEN_ISSUER", "reviewer-service"),
        keyring_path=Path(
            os.environ.get(
                "REVIEWER_TOKEN_KEYRING_FILE",
                issuer_dir / "reviewer_token_keyring.json",
            )
        ),
        revocation_path=Path(
            os.environ.get(
                "REVIEWER_TOKEN_REVOCATION_FILE",
                issuer_dir / "reviewer_token_revocations.json",
            )
        ),
        admin_token=str(
            os.environ.get(
                "REVIEWER_ISSUER_ADMIN_TOKEN",
                os.environ.get("REVIEWER_TOKEN_SHARED_SECRET", "local-dev-issuer-admin-token"),
            )
        ),
        default_ttl_seconds=int(os.environ.get("REVIEWER_TOKEN_DEFAULT_TTL", "900") or 900),
    )


def create_reviewer_service_app(base_dir: Optional[Path] = None) -> FastAPI:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    app = FastAPI(title="Reviewer Service API")
    rate_limit_state: Dict[str, list[float]] = {}

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time()
        response = await call_next(request)
        logger.info(
            "reviewer_service_access method=%s path=%s status=%s duration_ms=%s client=%s",
            request.method,
            request.url.path,
            response.status_code,
            int((time() - started) * 1000),
            request.client.host if request.client else "unknown",
        )
        return response

    def _enforce_rate_limit(identity: str) -> None:
        service = _build_service(resolved_base_dir)
        limit = int(service.config.get("rate_limit_rpm", 60) or 60)
        now = time()
        window_start = now - 60.0
        timestamps = [ts for ts in rate_limit_state.get(identity, []) if ts >= window_start]
        if len(timestamps) >= limit:
            raise HTTPException(status_code=429, detail="Reviewer service rate limit exceeded")
        timestamps.append(now)
        rate_limit_state[identity] = timestamps

    def _auth_context(
        reviewer_token: Optional[str],
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        reviewer_roles: Optional[str],
    ):
        service = _build_service(resolved_base_dir)
        try:
            auth = service.authenticate(
                token=str(reviewer_token or ""),
                reviewer_id=reviewer_id,
                tenant_id=tenant_id,
                roles=[role.strip() for role in (reviewer_roles or "reviewer").split(",") if role.strip()],
            )
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _enforce_rate_limit(f"{auth.reviewer_id}:{auth.tenant_id}")
        return service, auth

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "service": "reviewer-service"}

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        service = _build_service(resolved_base_dir)
        health = service.health()
        if health.get("status") not in {"ok", "ready"}:
            raise HTTPException(status_code=503, detail=health)
        return {**health, "status": "ready"}

    @app.get("/issuer/health")
    async def issuer_health() -> Dict[str, Any]:
        issuer = _build_token_issuer(resolved_base_dir)
        return issuer.health()

    @app.get("/reviews")
    async def list_reviews(
        status: str = "pending",
        include_resolved: bool = False,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        service, auth = _auth_context(
            x_reviewer_token,
            x_reviewer_id,
            x_tenant_id,
            x_reviewer_roles,
        )
        return service.list_reviews(
            auth,
            status=status if status else None,
            include_resolved=include_resolved,
        )

    @app.get("/reviews/summary")
    async def review_summary(
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        service, auth = _auth_context(
            x_reviewer_token,
            x_reviewer_id,
            x_tenant_id,
            x_reviewer_roles,
        )
        return service.get_summary(auth)

    @app.post("/reviews/submit")
    async def submit_review(
        payload: ReviewerDecisionPayload,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        service, auth = _auth_context(
            x_reviewer_token,
            x_reviewer_id,
            x_tenant_id,
            x_reviewer_roles,
        )
        return service.submit_review(auth, payload.model_dump(exclude_none=True))

    @app.post("/issuer/tokens")
    async def issue_reviewer_token(
        payload: ReviewerTokenIssuePayload,
        x_issuer_admin_token: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        issuer = _build_token_issuer(resolved_base_dir)
        try:
            return issuer.issue_token(
                admin_token=str(x_issuer_admin_token or ""),
                reviewer_id=payload.reviewer_id,
                tenant_ids=payload.tenant_ids,
                roles=payload.roles,
                ttl_seconds=payload.ttl_seconds,
                metadata=payload.metadata,
            )
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/issuer/tokens/rotate")
    async def rotate_reviewer_token_key(
        x_issuer_admin_token: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        issuer = _build_token_issuer(resolved_base_dir)
        try:
            return issuer.rotate_signing_key(admin_token=str(x_issuer_admin_token or ""))
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/issuer/tokens/revoke")
    async def revoke_reviewer_token(
        payload: ReviewerTokenRevokePayload,
        x_issuer_admin_token: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        issuer = _build_token_issuer(resolved_base_dir)
        try:
            return issuer.revoke_token(
                admin_token=str(x_issuer_admin_token or ""),
                token=payload.token,
                jti=payload.jti,
                reason=payload.reason,
            )
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/forensics/search")
    async def forensics_search(
        query: str = "",
        severity: Optional[str] = None,
        regression_label: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 50,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id, x_reviewer_roles)
        return search_observability_artifacts(
            resolved_base_dir,
            query=query,
            severity=severity,
            regression_label=regression_label,
            run_id=run_id,
            limit=limit,
        )

    @app.get("/forensics/triage")
    async def forensics_triage(
        limit: int = 50,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id, x_reviewer_roles)
        return build_observability_triage_queue(resolved_base_dir, limit=limit)

    @app.get("/forensics/diff/{run_id}")
    async def forensics_diff(
        run_id: str,
        compare_to_run_id: Optional[str] = None,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id, x_reviewer_roles)
        try:
            return build_artifact_diff_view(
                resolved_base_dir,
                run_id=run_id,
                compare_to_run_id=compare_to_run_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/forensics/clusters/{cluster_id}")
    async def forensics_cluster_drilldown(
        cluster_id: str,
        x_reviewer_token: Optional[str] = Header(default=None),
        x_reviewer_id: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
        x_reviewer_roles: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _auth_context(x_reviewer_token, x_reviewer_id, x_tenant_id, x_reviewer_roles)
        try:
            return get_issue_cluster_drilldown(resolved_base_dir, cluster_id=cluster_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app