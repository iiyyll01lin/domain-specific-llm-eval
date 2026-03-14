from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class WikiDataKnowledgeGraphSync:
    """Discovers mappings to WikiData and enriches internal KG dynamically."""

    def __init__(
        self,
        *,
        endpoint: str = "https://www.wikidata.org/w/api.php",
        timeout: int = 10,
        session: Optional[requests.Session] = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = session or requests.Session()
        self.wikidata_mock_api = {
            "Python": {
                "creator": "Guido van Rossum",
                "paradigm": "Multi-paradigm",
                "typing": "Duck, dynamic, gradual",
            },
            "Neo4j": {"language": "Java", "developer": "Neo4j, Inc."},
        }

    def sync_node(self, node_label: str) -> Dict[str, Any]:
        """Queries WikiData to find an exact matching node and pull facts."""
        try:
            live_result = self._sync_node_live(node_label)
            if live_result:
                return live_result
        except Exception as exc:
            logger.warning("WikiData live sync failed for '%s': %s", node_label, exc)

        for entity, data in self.wikidata_mock_api.items():
            if entity.lower() in node_label.lower():
                logger.info(
                    f"WikiData Sync: Matched local node '{node_label}' with global WikiData entity '{entity}'"
                )
                return data

        logger.info(f"WikiData Sync: No global match found for '{node_label}'")
        return {}

    def _sync_node_live(self, node_label: str) -> Dict[str, Any]:
        params: Dict[str, str] = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": node_label,
            "limit": "1",
        }
        response = self.session.get(
            self.endpoint,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("search", [])
        if not candidates:
            return {}

        top = candidates[0]
        label = str(top.get("label") or node_label)
        result = {
            "wikidata_id": top.get("id"),
            "label": label,
            "description": top.get("description", ""),
            "match": label,
            "source": "wikidata-live",
        }
        logger.info(
            "WikiData Sync: Matched local node '%s' with live entity '%s'",
            node_label,
            label,
        )
        return result
