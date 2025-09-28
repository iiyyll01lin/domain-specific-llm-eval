import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.processing.stages import SentenceSegment, SentenceTokenizer  # noqa: E402


def _segment_texts(segments: List[SentenceSegment]) -> List[str]:
    return [segment.text for segment in segments]


def test_tokenizer_handles_mixed_languages():
    text = "Hello world. 第二句測試。Third line!\n第四行？ Mixed feelings.\n\nLast one."
    tokenizer = SentenceTokenizer()

    segments = tokenizer.tokenize(text, mime_type="text/plain")

    assert _segment_texts(segments) == [
        "Hello world.",
        "第二句測試。",
        "Third line!",
        "第四行？",
        "Mixed feelings.",
        "Last one.",
    ]
    assert [segment.length for segment in segments] == [12, 6, 11, 4, 15, 9]


def test_tokenizer_supports_pdf_mime():
    text = "First pdf sentence. 第二句。"
    tokenizer = SentenceTokenizer()

    segments = tokenizer.tokenize(text, mime_type="application/pdf")

    assert _segment_texts(segments) == ["First pdf sentence.", "第二句。"]


def test_tokenizer_fallback_for_unsupported_mime(caplog: pytest.LogCaptureFixture):
    text = "Fallback example goes here."
    tokenizer = SentenceTokenizer()

    with caplog.at_level("INFO"):
        segments = tokenizer.tokenize(text, mime_type="application/octet-stream")

    assert _segment_texts(segments) == ["Fallback example goes here."]
    assert any("Tokenizer fallback" in message for message in caplog.text.splitlines())


def test_tokenizer_empty_text_returns_empty_list():
    tokenizer = SentenceTokenizer()

    assert tokenizer.tokenize("    \n\n", mime_type="text/plain") == []
