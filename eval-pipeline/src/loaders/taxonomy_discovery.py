from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ZeroShotTaxonomyDiscoverer:
    def __init__(self, llm_endpoint: str = "local") -> None:
        self.llm_endpoint = llm_endpoint

    def extract_ontology(self, raw_csv_content: str) -> Dict[str, Any]:
        logger.info("Analyzing unlabelled CSV for zero-shot taxonomy discovery...")
        reader = csv.DictReader(io.StringIO(raw_csv_content))
        rows = list(reader)
        headers = reader.fieldnames or []

        entities: List[str] = []
        relations: List[str] = []
        confidence: Dict[str, float] = {}

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
        if any(header in lowered_headers for header in ["patient", "patient_id", "mrn"]) and any(
            header in lowered_headers for header in ["policy", "policy_id", "policy_name"]
        ):
            relations.append("AFFECTS")

        if not entities and rows:
            entities.append("Record")
            confidence["Record"] = 0.5

        proposals = [
            {"entity": entity, "confidence": confidence.get(entity, 0.5)}
            for entity in entities
        ]

        return {
            "entities": entities,
            "relations": relations,
            "proposals": proposals,
            "headers": headers,
            "sample_value_counts": dict(value_tokens.most_common(5)),
        }
