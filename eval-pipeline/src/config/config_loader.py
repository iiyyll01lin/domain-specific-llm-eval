from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.config_manager import ConfigManager


class ConfigLoader:
    """Backward-compatible wrapper around ConfigManager."""

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        manager = ConfigManager(config_path or str(Path("config/pipeline_config.yaml")))
        return manager.load_config()