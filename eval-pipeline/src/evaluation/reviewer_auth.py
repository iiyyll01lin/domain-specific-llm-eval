from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class ReviewerPrincipal:
    reviewer_id: str
    tenant_ids: List[str]
    roles: List[str]
    auth_source: str
    claims: Dict[str, Any]


class ReviewerAuthSource(Protocol):
    def authenticate(
        self,
        *,
        token: str,
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[List[str]] = None,
    ) -> ReviewerPrincipal:
        ...

    def health(self) -> Dict[str, Any]:
        ...


class StaticReviewerAuthSource:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.auth_config = config.get("auth", {})
        self.reviewer_tokens = config.get("reviewer_tokens", {})

    def authenticate(
        self,
        *,
        token: str,
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[List[str]] = None,
    ) -> ReviewerPrincipal:
        expected_token = str(self.auth_config.get("api_token", "")).strip()
        if expected_token and not hmac.compare_digest(token, expected_token) and token not in self.reviewer_tokens:
            raise PermissionError("Invalid reviewer token")

        resolved_reviewer = str(
            reviewer_id or self.auth_config.get("default_reviewer_id", "reviewer")
        ).strip()
        resolved_tenant = str(
            tenant_id or self.auth_config.get("default_tenant_id", "default")
        ).strip()
        resolved_roles = list(roles or self.auth_config.get("default_roles", ["reviewer"]))

        token_binding = self.reviewer_tokens.get(token)
        if isinstance(token_binding, dict):
            resolved_reviewer = str(
                token_binding.get("reviewer_id", resolved_reviewer)
            ).strip()
            resolved_tenant = str(token_binding.get("tenant_id", resolved_tenant)).strip()
            resolved_roles = list(token_binding.get("roles", resolved_roles))

        return ReviewerPrincipal(
            reviewer_id=resolved_reviewer,
            tenant_ids=[resolved_tenant],
            roles=resolved_roles,
            auth_source="static-token",
            claims={"reviewer_id": resolved_reviewer, "tenant_ids": [resolved_tenant], "roles": resolved_roles},
        )

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "auth_source": "static-token"}


class FileBackedReviewerAuthSource:
    def __init__(self, principal_file: Path) -> None:
        self.principal_file = Path(principal_file)

    def _load_principals(self) -> Dict[str, Any]:
        if not self.principal_file.exists():
            raise PermissionError(f"Principal file not found: {self.principal_file}")
        return json.loads(self.principal_file.read_text(encoding="utf-8"))

    def authenticate(
        self,
        *,
        token: str,
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[List[str]] = None,
    ) -> ReviewerPrincipal:
        payload = self._load_principals()
        token_map = payload.get("tokens", {})
        token_payload = token_map.get(token)
        if not isinstance(token_payload, dict):
            raise PermissionError("Invalid reviewer token")

        resolved_reviewer = str(
            reviewer_id or token_payload.get("reviewer_id", "reviewer")
        ).strip()
        tenant_ids = [str(value).strip() for value in token_payload.get("tenant_ids", []) if str(value).strip()]
        selected_tenant = str(tenant_id or token_payload.get("default_tenant_id", tenant_ids[0] if tenant_ids else "default")).strip()
        if selected_tenant not in tenant_ids:
            raise PermissionError(f"Reviewer '{resolved_reviewer}' is not a member of tenant '{selected_tenant}'")

        resolved_roles = list(roles or token_payload.get("roles", ["reviewer"]))
        return ReviewerPrincipal(
            reviewer_id=resolved_reviewer,
            tenant_ids=tenant_ids,
            roles=resolved_roles,
            auth_source="principal-file",
            claims=token_payload,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.principal_file.exists() else "degraded",
            "auth_source": "principal-file",
            "principal_file": str(self.principal_file),
        }


class InternalTokenIssuerAuthSource:
    def __init__(self, issuer: str, secret: str) -> None:
        self.issuer = issuer
        self.secret = secret.encode("utf-8")

    def _decode_token(self, token: str) -> Dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "rtk":
            raise PermissionError("Invalid internal reviewer token format")

        payload_segment = parts[1].encode("utf-8")
        signature = parts[2]
        expected_signature = hmac.new(self.secret, payload_segment, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise PermissionError("Invalid internal reviewer token signature")

        padded_segment = payload_segment + b"=" * ((4 - len(payload_segment) % 4) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded_segment).decode("utf-8"))
        expires_at = int(payload.get("exp", 0) or 0)
        if expires_at and expires_at < int(time.time()):
            raise PermissionError("Internal reviewer token has expired")
        if str(payload.get("iss", "")) != self.issuer:
            raise PermissionError("Unexpected reviewer token issuer")
        return payload

    def authenticate(
        self,
        *,
        token: str,
        reviewer_id: Optional[str],
        tenant_id: Optional[str],
        roles: Optional[List[str]] = None,
    ) -> ReviewerPrincipal:
        payload = self._decode_token(token)
        resolved_reviewer = str(reviewer_id or payload.get("sub", payload.get("reviewer_id", "reviewer"))).strip()
        tenant_ids = [str(value).strip() for value in payload.get("tenant_ids", []) if str(value).strip()]
        selected_tenant = str(tenant_id or payload.get("tenant_id", tenant_ids[0] if tenant_ids else "default")).strip()
        if tenant_ids and selected_tenant not in tenant_ids:
            raise PermissionError(f"Reviewer '{resolved_reviewer}' is not a member of tenant '{selected_tenant}'")
        resolved_roles = list(roles or payload.get("roles", ["reviewer"]))
        return ReviewerPrincipal(
            reviewer_id=resolved_reviewer,
            tenant_ids=tenant_ids or [selected_tenant],
            roles=resolved_roles,
            auth_source="internal-token",
            claims=payload,
        )

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "auth_source": "internal-token", "issuer": self.issuer}


def build_reviewer_auth_source(config: Dict[str, Any]) -> ReviewerAuthSource:
    auth_source_config = config.get("auth_source", {})
    source_type = str(auth_source_config.get("type", "static-token")).strip() or "static-token"
    if source_type == "principal-file":
        principal_file = Path(str(auth_source_config.get("principal_file", ""))).expanduser()
        return FileBackedReviewerAuthSource(principal_file)
    if source_type == "internal-token":
        issuer = str(auth_source_config.get("issuer", "reviewer-service")).strip()
        secret = str(auth_source_config.get("shared_secret", "")).strip()
        if not secret:
            raise ValueError("internal-token auth source requires shared_secret")
        return InternalTokenIssuerAuthSource(issuer=issuer, secret=secret)
    return StaticReviewerAuthSource(config)