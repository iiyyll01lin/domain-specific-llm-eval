from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional
from unittest import mock

import pytest
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.common.errors import ServiceError
from services.common.storage.object_store import compute_checksum
from services.ingestion.repository import DocumentRecord
from services.processing.stages import ExtractedDocument, TextExtractionConfig, TextExtractor


@dataclass
class StubObjectStore:
    payloads: Dict[str, bytes]

    def download_bytes(
        self,
        bucket: Optional[str],
        key: str,
        expected_checksum: Optional[str] = None,
    ) -> bytes:
        if key not in self.payloads:
            raise KeyError(key)
        payload = self.payloads[key]
        if expected_checksum and compute_checksum(payload) != expected_checksum:
            raise AssertionError("Checksum mismatch in stub store")
        return payload


def _build_document(storage_key: str, payload: bytes) -> DocumentRecord:
    checksum = compute_checksum(payload)
    return DocumentRecord(
        document_id="doc-123",
        km_id="KM-001",
        version="v1",
        checksum=checksum,
        storage_key=storage_key,
        size_bytes=len(payload),
        created_at="2025-09-26T00:00:00Z",
        updated_at="2025-09-26T00:00:00Z",
    )


def test_plain_text_extraction_and_normalization():
    payload = "  Hello \tWorld!  \r\n\r\n Next line with\u3000wide space.  ".encode("utf-8")
    document = _build_document("documents/doc-123.txt", payload)
    extractor = TextExtractor(
        object_store=StubObjectStore({document.storage_key: payload}),
        config=TextExtractionConfig(bucket="unit-test"),
    )

    result = extractor.extract(document)

    assert isinstance(result, ExtractedDocument)
    assert result.mime_type == "text/plain"
    assert result.text == "Hello World!\nNext line with wide space."
    assert result.metadata == {"km_id": "KM-001", "version": "v1"}


def test_pdf_extraction_uses_pdfminer():
    payload = b"%PDF-1.4...binary"
    document = _build_document("documents/doc-123.pdf", payload)
    extractor = TextExtractor(
        object_store=StubObjectStore({document.storage_key: payload}),
        config=TextExtractionConfig(bucket=None),
    )

    with mock.patch("services.processing.stages.extract.pdf_extract_text", return_value="Hello PDF!  \n\nSecond line") as mocked:
        result = extractor.extract(document)

    mocked.assert_called_once()
    assert result.mime_type == "application/pdf"
    assert result.text == "Hello PDF!\nSecond line"


def test_pdf_extraction_not_allowed_raises_service_error():
    payload = b"%PDF-1.4...binary"
    document = _build_document("documents/doc-123.pdf", payload)
    extractor = TextExtractor(
        object_store=StubObjectStore({document.storage_key: payload}),
    )

    with mock.patch(
        "services.processing.stages.extract.pdf_extract_text",
        side_effect=PDFTextExtractionNotAllowed("forbidden"),
    ):
        with pytest.raises(ServiceError) as exc:
            extractor.extract(document)
    assert exc.value.error_code == "pdf_text_not_allowed"


def test_binary_document_raises_unsupported_type():
    payload = bytes([0, 159, 10, 0, 45, 0, 88]) * 20
    document = _build_document("documents/doc-123.bin", payload)
    extractor = TextExtractor(
        object_store=StubObjectStore({document.storage_key: payload}),
    )

    with pytest.raises(ServiceError) as exc:
        extractor.extract(document)
    assert exc.value.error_code == "unsupported_mime_type"
