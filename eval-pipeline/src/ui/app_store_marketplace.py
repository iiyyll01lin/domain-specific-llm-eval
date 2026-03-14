from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedAppStore:
    def __init__(self) -> None:
        self.registry: List[Dict[str, str]] = [
            {
                "id": "legal-tax-suite",
                "name": "Legal Tax Policy Prompts",
                "author": "GovTech",
                "version": "1.0.0",
                "trust": "verified",
                "dependencies": "base-eval",
            },
            {
                "id": "med-compliance-101",
                "name": "Medical HIPAA QA Runbook",
                "author": "HealthOrg",
                "version": "2.1.0",
                "trust": "verified",
                "dependencies": "base-eval,phi-redaction",
            },
        ]
        self.installed: List[Dict[str, str]] = []

    def get_runbook_manifest(self, runbook_id: str) -> Optional[Dict[str, str]]:
        for runbook in self.registry:
            if runbook["id"] == runbook_id:
                return dict(runbook)
        return None

    def install_runbook(self, runbook_id: str) -> bool:
        manifest = self.get_runbook_manifest(runbook_id)
        if manifest is not None and manifest.get("trust") == "verified":
            self.installed.append(manifest)
            logger.info(
                "Successfully installed Runbook: %s into local Helm cluster.",
                manifest["name"],
            )
            return True
        logger.error(f"Runbook {runbook_id} not found.")
        return False
