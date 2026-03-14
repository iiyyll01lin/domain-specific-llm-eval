from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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
    ):
        self.server_url = server_url
        self.signing_secret = signing_secret
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
        payload = {
            "node_id": envelope.node_id,
            "score": round(float(envelope.score), 6),
            "sample_count": int(envelope.sample_count),
            "tenant": envelope.tenant,
        }
        return envelope.signature == self._sign_payload(payload)

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
            "envelopes": [asdict(envelope) for envelope in valid_envelopes],
        }
