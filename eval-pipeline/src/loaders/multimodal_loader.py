import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MultimodalDocumentLoader:
    def __init__(self, enable_ocr: bool = True, enable_audio: bool = True) -> None:
        self.enable_ocr = enable_ocr
        self.enable_audio = enable_audio

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Loading multimodal document from {file_path}")
        docs: List[Dict[str, Any]] = [{"type": "text", "content": "Text content"}]
        if self.enable_ocr:
            docs.append({"type": "image", "content": "Bounding Box: [0,0,100,100]"})
        if self.enable_audio:
            docs.append({"type": "audio", "content": "Transcript: audio text"})
        return docs
