import logging

import pytest

from services.processing.stages.tokenizer import SentenceTokenizer


def test_tokenizer_fallbacks_to_single_segment_for_unsupported_mime(caplog: pytest.LogCaptureFixture) -> None:
    tokenizer = SentenceTokenizer()
    caplog.set_level(logging.INFO)

    text = "Hello world! 你好世界！"
    segments = tokenizer.tokenize(text, mime_type="application/vnd.ms-word")

    assert len(segments) == 1
    assert segments[0].text == text
    assert "Tokenizer fallback" in caplog.text


def test_tokenizer_handles_mixed_language_sentences() -> None:
    tokenizer = SentenceTokenizer()
    text = "Hello world! 你好世界！這是一個測試。\nFinal line without punctuation"

    segments = tokenizer.tokenize(text)

    assert [segment.text for segment in segments] == [
        "Hello world!",
        "你好世界！",
        "這是一個測試。",
        "Final line without punctuation",
    ]
