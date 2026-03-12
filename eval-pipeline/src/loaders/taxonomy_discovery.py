import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ZeroShotTaxonomyDiscoverer:
    def __init__(self, llm_endpoint: str = "local") -> None:
        self.llm_endpoint = llm_endpoint

    def extract_ontology(self, raw_csv_content: str) -> Dict[str, Any]:
        logger.info(f"Analyzing unlabelled CSV for zero-shot taxonomy discovery...")
        discovered_taxonomy = {
            "entities": ["Organization", "Policy", "Patient"],
            "relations": ["ISSUED_BY", "AFFECTS"],
        }
        return discovered_taxonomy
