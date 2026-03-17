from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeResultEnvelope:
    node_id: str
    score: float
    sample_count: int
    tenant: str
    signature: str


class FederatedLearningClient:
    """Manages Edge Tier local evaluation and gradient/RLHF scores aggregation."""

    def __init__(
        self,
        server_url: str = "http://central-parameter-server.internal",
        *,
        signing_secret: str = "federated-demo-secret",
        session: Optional[requests.Session] = None,
        timeout: int = 5,
        spool_dir: str | Path = "outputs/federated_spool",
        accepted_tenants: Optional[Sequence[str]] = None,
    ):
        self.server_url = server_url
        self.signing_secret = signing_secret
        self.session = session or requests.Session()
        self.timeout = timeout
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.spool_dir / "federated_audit.jsonl"
        self.accepted_tenants = {str(item) for item in (accepted_tenants or [])}
        self.is_connected = False

    def connect(self) -> None:
        logger.info(f"Connecting to Federated Parameter Server at {self.server_url}")
        self.is_connected = True

    def create_envelope(
        self,
        *,
        node_id: str,
        score: float,
        sample_count: int,
        tenant: str,
    ) -> EdgeResultEnvelope:
        normalized_score = round(float(score), 6)
        normalized_count = int(sample_count)
        payload = {
            "node_id": node_id,
            "score": normalized_score,
            "sample_count": normalized_count,
            "tenant": tenant,
        }
        signature = self._sign_payload(payload)
        return EdgeResultEnvelope(
            node_id=node_id,
            score=normalized_score,
            sample_count=normalized_count,
            tenant=tenant,
            signature=signature,
        )

    def validate_envelope(self, envelope: EdgeResultEnvelope) -> bool:
        if self.accepted_tenants and envelope.tenant not in self.accepted_tenants:
            return False
        payload = {
            "node_id": envelope.node_id,
            "score": round(float(envelope.score), 6),
            "sample_count": int(envelope.sample_count),
            "tenant": envelope.tenant,
        }
        return envelope.signature == self._sign_payload(payload)

    def _append_audit_event(self, payload: Dict[str, Any]) -> None:
        event = dict(payload)
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(self.audit_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _sign_payload(self, payload: Dict[str, Any]) -> str:
        material = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.signing_secret}:{material}".encode("utf-8")).hexdigest()

    def aggregate_gradients(
        self, local_scores: List[Dict[str, Any] | EdgeResultEnvelope]
    ) -> Dict[str, Any]:
        """Aggregate RLHF scores to parameter server to preserve PII."""
        if not self.is_connected:
            self.connect()
        logger.info(f"Aggregating {len(local_scores)} local scores to {self.server_url}")

        envelopes: List[EdgeResultEnvelope] = []
        for item in local_scores:
            if isinstance(item, EdgeResultEnvelope):
                envelopes.append(item)
            else:
                envelopes.append(
                    self.create_envelope(
                        node_id=str(item.get("node_id", f"edge-{len(envelopes)+1}")),
                        score=float(item.get("score", 0.0)),
                        sample_count=int(item.get("sample_count", 1)),
                        tenant=str(item.get("tenant", "default")),
                    )
                )

        valid_envelopes = [envelope for envelope in envelopes if self.validate_envelope(envelope)]
        total_samples = sum(envelope.sample_count for envelope in valid_envelopes)
        weighted_score = (
            sum(envelope.score * envelope.sample_count for envelope in valid_envelopes) / total_samples
            if total_samples
            else 0.0
        )
        tenants = sorted({envelope.tenant for envelope in valid_envelopes})

        return {
            "status": "success",
            "aggregated_count": len(valid_envelopes),
            "rejected_count": len(envelopes) - len(valid_envelopes),
            "weighted_score": round(weighted_score, 6),
            "total_samples": total_samples,
            "tenants": tenants,
            "trust_policy": sorted(self.accepted_tenants),
            "envelopes": [asdict(envelope) for envelope in valid_envelopes],
        }

    def submit_aggregation(
        self, local_scores: List[Dict[str, Any] | EdgeResultEnvelope]
    ) -> Dict[str, Any]:
        aggregated = self.aggregate_gradients(local_scores)
        try:
            response = self.session.post(
                f"{self.server_url.rstrip('/')}/aggregate",
                json=aggregated,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json() if hasattr(response, "json") else {}
            aggregated["submitted"] = True
            aggregated["server_response"] = payload if isinstance(payload, dict) else {}
            self._append_audit_event(
                {
                    "event": "submitted",
                    "server_url": self.server_url,
                    "submitted": True,
                    "aggregated_count": aggregated.get("aggregated_count", 0),
                    "tenants": aggregated.get("tenants", []),
                }
            )
            return aggregated
        except Exception:
            spool_path = self.spool_dir / "pending_federated_submission.json"
            spool_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
            aggregated["submitted"] = False
            aggregated["spool_path"] = str(spool_path)
            self._append_audit_event(
                {
                    "event": "spooled",
                    "server_url": self.server_url,
                    "submitted": False,
                    "spool_path": str(spool_path),
                    "aggregated_count": aggregated.get("aggregated_count", 0),
                    "tenants": aggregated.get("tenants", []),
                }
            )
            return aggregated

    def replay_spooled_submission(self) -> Optional[Dict[str, Any]]:
        spool_path = self.spool_dir / "pending_federated_submission.json"
        if not spool_path.exists():
            return None
        payload = json.loads(spool_path.read_text(encoding="utf-8"))
        response = self.session.post(
            f"{self.server_url.rstrip('/')}/aggregate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        spool_path.unlink(missing_ok=True)
        self._append_audit_event(
            {
                "event": "replayed",
                "server_url": self.server_url,
                "submitted": True,
                "aggregated_count": payload.get("aggregated_count", 0),
                "tenants": payload.get("tenants", []),
            }
        )
        return payload
