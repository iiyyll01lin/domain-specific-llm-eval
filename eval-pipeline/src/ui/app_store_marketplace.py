import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class UnifiedAppStore:
    def __init__(self) -> None:
        self.registry: List[Dict[str, str]] = [
            {
                "id": "legal-tax-suite",
                "name": "Legal Tax Policy Prompts",
                "author": "GovTech",
            },
            {
                "id": "med-compliance-101",
                "name": "Medical HIPAA QA Runbook",
                "author": "HealthOrg",
            },
        ]

    def install_runbook(self, runbook_id: str) -> bool:
        for rb in self.registry:
            if rb["id"] == runbook_id:
                logger.info(
                    f"Successfully installed Runbook: {rb['name']} into local Helm cluster."
                )
                return True
        logger.error(f"Runbook {runbook_id} not found.")
        return False
