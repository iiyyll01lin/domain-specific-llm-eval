from __future__ import annotations

import csv
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


class ZeroShotTaxonomyDiscoverer:
    def __init__(
        self,
        llm_endpoint: str = "local",
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 5,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.session = session or requests.Session()
        self.timeout = timeout

    def extract_ontology(self, raw_csv_content: str) -> Dict[str, Any]:
        logger.info("Analyzing unlabelled CSV for zero-shot taxonomy discovery...")
        reader = csv.DictReader(io.StringIO(raw_csv_content))
        rows = list(reader)
        headers = reader.fieldnames or []

        entities: List[str] = []
        relations: List[str] = []
        confidence: Dict[str, float] = {}
        hierarchy: List[Dict[str, str]] = []
        inferred_relations: List[Dict[str, Any]] = []

        lowered_headers = [header.lower() for header in headers]
        if any(header in lowered_headers for header in ["patient", "patient_id", "mrn"]):
            entities.append("Patient")
            confidence["Patient"] = 0.9
        if any(header in lowered_headers for header in ["policy", "policy_id", "policy_name"]):
            entities.append("Policy")
            confidence["Policy"] = 0.85
        if any(header in lowered_headers for header in ["organization", "org", "company", "department"]):
            entities.append("Organization")
            confidence["Organization"] = 0.8
        if any(header in lowered_headers for header in ["device", "error_code", "asset_id", "machine"]):
            entities.append("Asset")
            confidence["Asset"] = 0.7

        value_tokens: Counter[str] = Counter()
        for row in rows:
            for value in row.values():
                token = str(value).strip()
                if token:
                    value_tokens[token.lower()] += 1

        if any("policy" in header for header in lowered_headers) and any(
            header in lowered_headers for header in ["organization", "org", "department"]
        ):
            relations.append("ISSUED_BY")
            hierarchy.append({"parent": "Organization", "child": "Policy"})
        if any(header in lowered_headers for header in ["patient", "patient_id", "mrn"]) and any(
            header in lowered_headers for header in ["policy", "policy_id", "policy_name"]
        ):
            relations.append("AFFECTS")
            hierarchy.append({"parent": "Patient", "child": "Policy"})
        if any(header in lowered_headers for header in ["device", "error_code", "asset_id", "machine"]) and any(
            header in lowered_headers for header in ["organization", "org", "department"]
        ):
            relations.append("OWNED_BY")
            hierarchy.append({"parent": "Organization", "child": "Asset"})

        if not entities and rows:
            entities.append("Record")
            confidence["Record"] = 0.5

        proposals = [
            {"entity": entity, "confidence": confidence.get(entity, 0.5)}
            for entity in entities
        ]

        non_empty_headers = [header for header in headers if header]
        for left in non_empty_headers:
            for right in non_empty_headers:
                if left == right:
                    continue
                overlap = self._header_token_overlap(left, right)
                if overlap <= 0:
                    continue
                inferred_relations.append(
                    {
                        "from": left,
                        "to": right,
                        "confidence": round(min(0.95, 0.45 + (overlap * 0.25)), 2),
                    }
                )

        return {
            "entities": sorted(set(entities)),
            "relations": sorted(set(relations)),
            "proposals": proposals,
            "headers": headers,
            "hierarchy": hierarchy,
            "relation_candidates": inferred_relations[:10],
            "sample_value_counts": dict(value_tokens.most_common(5)),
        }

    def _header_token_overlap(self, left: str, right: str) -> int:
        left_tokens: Set[str] = {token for token in left.lower().replace("-", "_").split("_") if token}
        right_tokens: Set[str] = {token for token in right.lower().replace("-", "_").split("_") if token}
        return len(left_tokens & right_tokens)

    def extract_ontology_from_csv_file(
        self,
        csv_path: str | Path,
        *,
        persist_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        payload = self.extract_ontology(Path(csv_path).read_text(encoding="utf-8"))
        payload["source_file"] = str(csv_path)
        payload["backend_used"] = False

        backend_enrichment = self._request_backend_enrichment(payload)
        if backend_enrichment:
            payload["backend_used"] = True
            payload["backend_enrichment"] = backend_enrichment
            payload["entities"] = sorted(
                set(payload["entities"]) | set(backend_enrichment.get("entities", []))
            )
            payload["relations"] = sorted(
                set(payload["relations"]) | set(backend_enrichment.get("relations", []))
            )
            payload["relation_candidates"] = list(payload.get("relation_candidates", [])) + [
                {
                    "from": "backend",
                    "to": relation,
                    "confidence": 0.9,
                }
                for relation in backend_enrichment.get("relations", [])
            ]

        if persist_path is not None:
            resolved = Path(persist_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["persist_path"] = str(resolved)
        return payload

    def approve_taxonomy_proposal(
        self,
        proposal: Dict[str, Any],
        *,
        approved_path: str | Path,
    ) -> str:
        approved_payload = dict(proposal)
        approved_payload["status"] = "approved"
        path = Path(approved_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(approved_payload, indent=2), encoding="utf-8")
        return str(path)

    def _request_backend_enrichment(
        self, ontology_payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self.llm_endpoint in {"", "local"}:
            return None
        try:
            response = self.session.post(
                self.llm_endpoint,
                json={
                    "headers": ontology_payload.get("headers", []),
                    "sample_value_counts": ontology_payload.get("sample_value_counts", {}),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return {
                    "entities": list(payload.get("entities", [])),
                    "relations": list(payload.get("relations", [])),
                }
        except Exception:
            return None
        return None
