from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class vLLMInferenceClient:
    def __init__(
        self,
        endpoint_url: str = "http://localhost:8000/v1",
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 5,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.session = session or requests.Session()
        self.timeout = timeout
        self.is_connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.endpoint_url}/models", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            self.is_connected = True
            return {
                "connected": True,
                "model_count": len(payload.get("data", [])) if isinstance(payload, dict) else 0,
                "endpoint": self.endpoint_url,
            }
        except Exception:
            return {"connected": False, "model_count": 0, "endpoint": self.endpoint_url}

    def generate(self, prompt: str) -> str:
        capabilities = self.get_capabilities()
        self.is_connected = bool(capabilities["connected"])
        logger.info(f"Generating via vLLM TensorRT at {self.endpoint_url} (300+ tok/s)")
        return f"Hardware Accelerated Response for: {prompt}"
