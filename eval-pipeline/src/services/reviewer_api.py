from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.ui.reviewer_actions import _build_service


class ReviewerDecisionPayload(BaseModel):
    review_id: Optional[str] = None
    index: Optional[int] = None
    question: str = Field(default="")
    approved: bool = Field(default=True)
    score: Optional[float] = None
    notes: str = Field(default="")
    reviewer: str = Field(default="reviewer")
    resolution: str = Field(default="resolved")


def create_reviewer_service_app(base_dir: Optional[Path] = None) -> FastAPI:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    app = FastAPI(title="Reviewer Service API")

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
        return service, auth

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "service": "reviewer-service"}

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

    return app