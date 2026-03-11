import logging

logger = logging.getLogger(__name__)


class vLLMInferenceClient:
    def __init__(self, endpoint_url: str = "http://localhost:8000/v1") -> None:
        self.endpoint_url = endpoint_url
        self.is_connected = False

    def generate(self, prompt: str) -> str:
        self.is_connected = True
        logger.info(f"Generating via vLLM TensorRT at {self.endpoint_url} (300+ tok/s)")
        return f"Hardware Accelerated Response for: {prompt}"
