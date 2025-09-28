from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional

from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFSyntaxError

from services.common.errors import ServiceError
from services.common.storage.object_store import ObjectStoreClient, compute_checksum
from services.ingestion.repository import DocumentRecord

logger = logging.getLogger(__name__)

_ZERO_WIDTH_PATTERN = re.compile("[\u200B\u200C\u200D\uFEFF]")
_DEFAULT_TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml")


@dataclass
class TextExtractionConfig:
    """Configuration for the text extraction stage."""

    bucket: Optional[str] = None
    strip_empty_lines: bool = True
    collapse_whitespace: bool = True
    minimum_text_length: int = 1


@dataclass
class ExtractedDocument:
    """Represents a normalized text payload derived from a raw document."""

    document_id: str
    checksum: str
    text: str
    byte_size: int
    mime_type: str
    metadata: Dict[str, str] = field(default_factory=dict)


class TextExtractor:
    """Downloads raw document content and produces normalized text."""

    def __init__(
        self,
        object_store: ObjectStoreClient,
        *,
        config: Optional[TextExtractionConfig] = None,
    ) -> None:
        self._object_store = object_store
        self._config = config or TextExtractionConfig()

    def extract(
        self,
        document: DocumentRecord,
        *,
        expected_checksum: Optional[str] = None,
    ) -> ExtractedDocument:
        payload = self._object_store.download_bytes(
            bucket=self._config.bucket,
            key=document.storage_key,
            expected_checksum=expected_checksum or document.checksum,
        )
        mime_type = self._detect_mime_type(payload, document.storage_key)
        raw_text = self._convert_to_text(payload, mime_type, document)
        normalized = self._normalize_text(raw_text)
        if len(normalized) < self._config.minimum_text_length:
            raise ServiceError(
                error_code="extraction_empty_text",
                message=f"Document {document.document_id} produced no extractable text",
                http_status=422,
            )
        checksum = compute_checksum(payload)
        logger.info(
            "🧾 extraction complete for %s (%s, %s bytes)",
            document.document_id,
            mime_type,
            len(payload),
        )
        return ExtractedDocument(
            document_id=document.document_id,
            checksum=checksum,
            text=normalized,
            byte_size=len(payload),
            mime_type=mime_type,
            metadata={
                "km_id": document.km_id,
                "version": document.version,
            },
        )

    def _detect_mime_type(self, payload: bytes, storage_key: str) -> str:
        if payload.startswith(b"%PDF"):
            return "application/pdf"
        lowered = storage_key.lower()
        if any(lowered.endswith(ext) for ext in _DEFAULT_TEXT_EXTENSIONS):
            return "text/plain"
        if self._looks_textual(payload):
            return "text/plain"
        return "application/octet-stream"

    def _looks_textual(self, payload: bytes) -> bool:
        if not payload:
            return False
        sample = payload[:1024]
        non_printable = 0
        for byte in sample:
            if byte in (9, 10, 13):  # tab / newline / carriage return
                continue
            if byte < 32 or byte == 127:
                non_printable += 1
        return (non_printable / max(1, len(sample))) < 0.1

    def _convert_to_text(self, payload: bytes, mime_type: str, document: DocumentRecord) -> str:
        if mime_type == "application/pdf":
            try:
                return pdf_extract_text(io.BytesIO(payload))
            except PDFTextExtractionNotAllowed as exc:
                raise ServiceError(
                    error_code="pdf_text_not_allowed",
                    message=f"PDF forbids text extraction for document {document.document_id}",
                    http_status=403,
                ) from exc
            except PDFSyntaxError as exc:
                raise ServiceError(
                    error_code="pdf_parse_failed",
                    message=f"Failed to parse PDF document {document.document_id}",
                    http_status=422,
                ) from exc
        if mime_type == "text/plain" or self._looks_textual(payload):
            return payload.decode("utf-8", errors="replace")
        raise ServiceError(
            error_code="unsupported_mime_type",
            message=f"Unsupported document type for extraction: {mime_type}",
            http_status=415,
        )

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _ZERO_WIDTH_PATTERN.sub("", normalized)
        lines = []
        for raw_line in normalized.split("\n"):
            line = raw_line
            if self._config.collapse_whitespace:
                line = re.sub(r"[ \t\f\v]+", " ", line)
            line = line.strip()
            if self._config.strip_empty_lines and not line:
                continue
            lines.append(line)
        return "\n".join(lines)


__all__ = ["ExtractedDocument", "TextExtractionConfig", "TextExtractor"]
