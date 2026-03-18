from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class ReviewerServiceClient:
    def __init__(
        self,
        service_url: str,
        api_token: str,
        reviewer_id: str,
        tenant_id: str,
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.api_token = api_token
        self.reviewer_id = reviewer_id
        self.tenant_id = tenant_id
        self.session = session or requests.Session()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Reviewer-Token": self.api_token,
            "X-Reviewer-Id": self.reviewer_id,
            "X-Tenant-Id": self.tenant_id,
        }

    def list_reviews(
        self,
        *,
        status: Optional[str] = "pending",
        include_resolved: bool = False,
    ) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.service_url}/reviews",
            params={
                "status": status,
                "include_resolved": include_resolved,
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_summary(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.service_url}/reviews/summary",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def submit_review(self, review_payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.service_url}/reviews/submit",
            json=review_payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()