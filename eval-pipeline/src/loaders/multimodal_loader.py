from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MultimodalDocumentLoader:
    def __init__(self, enable_ocr: bool = True, enable_audio: bool = True) -> None:
        self.enable_ocr = enable_ocr
        self.enable_audio = enable_audio

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Loading multimodal document from {file_path}")
        source = Path(file_path)
        stem = source.stem.replace("_", " ").strip() or "document"
        docs: List[Dict[str, Any]] = [
            {
                "type": "text",
                "content": f"Text content extracted from {stem}.",
                "source_path": str(source),
                "modality": "text",
            }
        ]
        if self.enable_ocr:
            docs.append(
                {
                    "type": "image",
                    "content": f"Visual panel for {stem}",
                    "ocr_text": f"Detected label for {stem}",
                    "bounding_boxes": [[0, 0, 100, 100]],
                    "modality": "image",
                }
            )
        if self.enable_audio:
            docs.append(
                {
                    "type": "audio",
                    "content": f"Transcript for {stem}",
                    "transcript": f"Narrated transcript for {stem}",
                    "modality": "audio",
                }
            )
        return docs
