from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

from src.evaluation.reviewer_auth import build_reviewer_auth_source


def _make_internal_token(payload: dict, secret: str) -> str:
    encoded_payload = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8").rstrip("=")
    signature = hmac.new(
        secret.encode("utf-8"),
        encoded_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"rtk.{encoded_payload}.{signature}"


def test_file_backed_auth_source_enforces_tenant_membership(tmp_path: Path) -> None:
    principal_file = tmp_path / "principals.json"
    principal_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "file-token": {
                        "reviewer_id": "alice",
                        "tenant_ids": ["tenant-a"],
                        "roles": ["reviewer"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    source = build_reviewer_auth_source(
        {"auth_source": {"type": "principal-file", "principal_file": str(principal_file)}}
    )
    principal = source.authenticate(
        token="file-token",
        reviewer_id=None,
        tenant_id="tenant-a",
        roles=None,
    )

    assert principal.reviewer_id == "alice"
    assert principal.auth_source == "principal-file"


def test_internal_token_auth_source_validates_signed_membership() -> None:
    secret = "unit-test-secret"
    source = build_reviewer_auth_source(
        {
            "auth_source": {
                "type": "internal-token",
                "issuer": "reviewer-service",
                "shared_secret": secret,
            }
        }
    )
    token = _make_internal_token(
        {
            "iss": "reviewer-service",
            "sub": "signed-reviewer",
            "tenant_ids": ["tenant-a", "tenant-b"],
            "roles": ["reviewer"],
            "exp": int(time.time()) + 300,
        },
        secret,
    )

    principal = source.authenticate(
        token=token,
        reviewer_id=None,
        tenant_id="tenant-b",
        roles=None,
    )

    assert principal.reviewer_id == "signed-reviewer"
    assert principal.tenant_ids == ["tenant-a", "tenant-b"]
    assert principal.auth_source == "internal-token"