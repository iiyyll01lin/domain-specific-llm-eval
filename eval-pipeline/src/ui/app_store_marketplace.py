from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedAppStore:
    BUILTIN_DEPENDENCIES = {"base-eval"}

    def __init__(
        self,
        *,
        manifest_dir: str | Path | None = None,
        install_dir: str | Path | None = None,
    ) -> None:
        self.manifest_dir = Path(manifest_dir) if manifest_dir is not None else None
        self.install_dir = Path(install_dir) if install_dir is not None else None
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
        self.sync_registry()

    def sync_registry(self) -> None:
        if self.manifest_dir is None or not self.manifest_dir.exists():
            return
        manifest_items: List[Dict[str, str]] = []
        for manifest_path in sorted(self.manifest_dir.glob("*.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("id"):
                    manifest_items.append({key: str(value) for key, value in payload.items()})
            except Exception:
                continue
        if manifest_items:
            self.registry = manifest_items

    def get_runbook_manifest(self, runbook_id: str) -> Optional[Dict[str, str]]:
        for runbook in self.registry:
            if runbook["id"] == runbook_id:
                return dict(runbook)
        return None

    def validate_dependencies(self, runbook_id: str) -> bool:
        manifest = self.get_runbook_manifest(runbook_id)
        if manifest is None:
            return False
        dependencies = [item.strip() for item in manifest.get("dependencies", "").split(",") if item.strip()]
        installed_ids = {item["id"] for item in self.installed}
        known_ids = {item["id"] for item in self.registry}
        return all(
            dependency in installed_ids
            or dependency in known_ids
            or dependency in self.BUILTIN_DEPENDENCIES
            for dependency in dependencies
        )

    def install_runbook(self, runbook_id: str) -> bool:
        manifest = self.get_runbook_manifest(runbook_id)
        if (
            manifest is not None
            and manifest.get("trust") == "verified"
            and self.validate_dependencies(runbook_id)
        ):
            self.installed.append(manifest)
            if self.install_dir is not None:
                self.install_dir.mkdir(parents=True, exist_ok=True)
                receipt = self.install_dir / f"{runbook_id}.json"
                receipt.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info(
                "Successfully installed Runbook: %s into local Helm cluster.",
                manifest["name"],
            )
            return True
        logger.error("Runbook %s could not be installed due to trust or dependency validation failure.", runbook_id)
        return False
