from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _b64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _b64url_decode(segment: str) -> bytes:
    padded = segment.encode("utf-8") + b"=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(padded)


class ReviewerTokenIssuerService:
    def __init__(
        self,
        *,
        issuer: str,
        keyring_path: Path,
        revocation_path: Path,
        admin_token: str,
        default_ttl_seconds: int = 900,
    ) -> None:
        self.issuer = str(issuer).strip() or "reviewer-service"
        self.keyring_path = Path(keyring_path)
        self.revocation_path = Path(revocation_path)
        self.admin_token = str(admin_token).strip()
        self.default_ttl_seconds = max(60, int(default_ttl_seconds))
        self.keyring_path.parent.mkdir(parents=True, exist_ok=True)
        self.revocation_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_state()

    def _ensure_state(self) -> None:
        if not self.keyring_path.exists():
            self._write_json(
                self.keyring_path,
                {
                    "issuer": self.issuer,
                    "active_kid": None,
                    "keys": [],
                    "rotations": [],
                },
            )
        if not self.revocation_path.exists():
            self._write_json(self.revocation_path, {"issuer": self.issuer, "revoked_tokens": []})
        keyring = self._load_keyring()
        if not keyring.get("active_kid"):
            self.rotate_signing_key(admin_token=self.admin_token)

    def _load_keyring(self) -> Dict[str, Any]:
        payload = json.loads(self.keyring_path.read_text(encoding="utf-8"))
        payload.setdefault("keys", [])
        payload.setdefault("rotations", [])
        payload.setdefault("issuer", self.issuer)
        return payload

    def _load_revocations(self) -> Dict[str, Any]:
        payload = json.loads(self.revocation_path.read_text(encoding="utf-8"))
        payload.setdefault("revoked_tokens", [])
        payload.setdefault("issuer", self.issuer)
        return payload

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _require_admin_token(self, admin_token: str) -> None:
        if not self.admin_token:
            raise PermissionError("Issuer admin token is not configured")
        if not hmac.compare_digest(str(admin_token).strip(), self.admin_token):
            raise PermissionError("Invalid issuer admin token")

    def issue_token(
        self,
        *,
        admin_token: str,
        reviewer_id: str,
        tenant_ids: List[str],
        roles: List[str],
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_admin_token(admin_token)
        keyring = self._load_keyring()
        active_key = next(
            (item for item in keyring["keys"] if item.get("kid") == keyring.get("active_kid")),
            None,
        )
        if active_key is None:
            raise RuntimeError("No active reviewer issuer signing key available")

        now = int(time.time())
        expires_in = max(60, int(ttl_seconds or self.default_ttl_seconds))
        payload = {
            "iss": self.issuer,
            "sub": str(reviewer_id).strip(),
            "reviewer_id": str(reviewer_id).strip(),
            "tenant_ids": [str(value).strip() for value in tenant_ids if str(value).strip()],
            "roles": [str(value).strip() for value in roles if str(value).strip()],
            "iat": now,
            "exp": now + expires_in,
            "jti": str(uuid.uuid4()),
            "kid": active_key["kid"],
            "metadata": metadata or {},
        }
        encoded_payload = _b64url_encode(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        signature = hmac.new(
            active_key["secret"].encode("utf-8"),
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        token = f"rtk.{encoded_payload}.{signature}"
        return {
            "token": token,
            "payload": payload,
            "expires_in": expires_in,
            "issuer": self.issuer,
            "kid": active_key["kid"],
        }

    def rotate_signing_key(
        self,
        *,
        admin_token: str,
        new_secret: Optional[str] = None,
        grace_period_seconds: int = 3600,
    ) -> Dict[str, Any]:
        self._require_admin_token(admin_token)
        keyring = self._load_keyring()
        now = int(time.time())
        active_kid = keyring.get("active_kid")
        if active_kid:
            for item in keyring["keys"]:
                if item.get("kid") == active_kid:
                    item["status"] = "grace"
                    item["grace_until"] = now + max(60, int(grace_period_seconds))

        new_key = {
            "kid": f"rk-{uuid.uuid4().hex[:12]}",
            "secret": str(new_secret or secrets.token_urlsafe(48)),
            "status": "active",
            "created_at": now,
            "grace_until": None,
        }
        keyring["keys"] = [item for item in keyring["keys"] if item.get("status") != "retired"]
        keyring["keys"].append(new_key)
        keyring["active_kid"] = new_key["kid"]
        keyring["rotations"].append(
            {
                "timestamp": now,
                "active_kid": new_key["kid"],
                "previous_kid": active_kid,
                "grace_period_seconds": max(60, int(grace_period_seconds)),
            }
        )
        self._write_json(self.keyring_path, keyring)
        return {
            "issuer": self.issuer,
            "active_kid": new_key["kid"],
            "rotated_at": now,
            "previous_kid": active_kid,
        }

    def revoke_token(
        self,
        *,
        admin_token: str,
        token: Optional[str] = None,
        jti: Optional[str] = None,
        reason: str = "manual_revocation",
    ) -> Dict[str, Any]:
        self._require_admin_token(admin_token)
        token_id = jti
        payload: Dict[str, Any] = {}
        if token:
            payload = self.inspect_token(token)
            token_id = str(payload.get("jti", "")).strip() or token_id
        if not token_id:
            raise ValueError("Token revocation requires token or jti")

        revocations = self._load_revocations()
        now = int(time.time())
        if not any(entry.get("jti") == token_id for entry in revocations["revoked_tokens"]):
            revocations["revoked_tokens"].append(
                {
                    "jti": token_id,
                    "reason": str(reason).strip() or "manual_revocation",
                    "revoked_at": now,
                    "expires_at": payload.get("exp"),
                    "reviewer_id": payload.get("reviewer_id") or payload.get("sub"),
                }
            )
            self._write_json(self.revocation_path, revocations)
        return {"revoked": True, "jti": token_id, "reason": reason, "revoked_at": now}

    def inspect_token(self, token: str) -> Dict[str, Any]:
        parts = str(token).split(".")
        if len(parts) != 3 or parts[0] != "rtk":
            raise PermissionError("Invalid internal reviewer token format")
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        return payload

    def health(self) -> Dict[str, Any]:
        keyring = self._load_keyring()
        revocations = self._load_revocations()
        return {
            "status": "ok",
            "issuer": self.issuer,
            "active_kid": keyring.get("active_kid"),
            "key_count": len(keyring.get("keys", [])),
            "revoked_token_count": len(revocations.get("revoked_tokens", [])),
        }