from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Set

logger = logging.getLogger(__name__)

_EN_ENDINGS: Set[str] = {".", "?", "!"}
_CJK_ENDINGS: Set[str] = {"。", "？", "！", "；"}
_CLOSING_QUOTES: Set[str] = {'"', "'", "”", "’", "）", ")", "】", "》", "」", "』", "]", "}"}
_SUPPORTED_MIME_TYPES: Set[str] = {"text/plain", "application/pdf"}


@dataclass(frozen=True)
class SentenceSegment:
    """Represents a contiguous span of normalized text."""

    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class SentenceTokenizer:
    """Simple multilingual-aware sentence tokenizer with fallback support."""

    def __init__(self, *, supported_mime_types: Iterable[str] | None = None) -> None:
        self._supported_mime_types: Set[str] = set(supported_mime_types or _SUPPORTED_MIME_TYPES)

    def tokenize(self, text: str, *, mime_type: str = "text/plain") -> List[SentenceSegment]:
        if not text.strip():
            return []
        if mime_type not in self._supported_mime_types:
            logger.info("Tokenizer fallback to single segment", extra={"mime_type": mime_type})
            fallback_segment = self._segment_from_bounds(text, 0, len(text))
            return [fallback_segment] if fallback_segment else []

        segments: List[SentenceSegment] = []
        start = 0
        length = len(text)
        for idx, char in enumerate(text):
            if char == "\n":
                segment = self._segment_from_bounds(text, start, idx)
                if segment:
                    segments.append(segment)
                start = idx + 1
                continue

            if self._is_sentence_terminator(char):
                end = idx + 1
                while end < length and text[end] in _CLOSING_QUOTES:
                    end += 1
                segment = self._segment_from_bounds(text, start, end)
                if segment:
                    segments.append(segment)
                start = end

        final_segment = self._segment_from_bounds(text, start, length)
        if final_segment:
            segments.append(final_segment)
        return segments

    def _is_sentence_terminator(self, char: str) -> bool:
        return char in _EN_ENDINGS or char in _CJK_ENDINGS

    def _segment_from_bounds(self, text: str, start: int, end: int) -> SentenceSegment | None:
        trimmed_start, trimmed_end = self._trim_bounds(text, start, end)
        if trimmed_start >= trimmed_end:
            return None
        return self._build_segment(text, trimmed_start, trimmed_end)

    def _build_segment(self, text: str, start: int, end: int) -> SentenceSegment:
        return SentenceSegment(text=text[start:end], start=start, end=end)

    @staticmethod
    def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end


__all__ = ["SentenceSegment", "SentenceTokenizer"]
