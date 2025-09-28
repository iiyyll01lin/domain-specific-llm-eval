from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from services.processing.stages.tokenizer import SentenceTokenizer


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration values controlling chunk assembly behaviour."""

    target_tokens: int = 512
    hard_max_tokens: int = 800
    overlap_tokens: int = 50
    histogram_bucket_size: int = 100
    use_tiktoken: bool = True

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.hard_max_tokens < self.target_tokens:
            raise ValueError("hard_max_tokens must be >= target_tokens")
        if self.hard_max_tokens <= 0:
            raise ValueError("hard_max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.histogram_bucket_size <= 0:
            raise ValueError("histogram_bucket_size must be positive")


@dataclass(frozen=True)
class ChunkCandidate:
    """Represents a chunk ready for downstream embedding and persistence."""

    sequence_index: int
    text: str
    token_count: int
    tokens: Tuple[Any, ...]


@dataclass(frozen=True)
class ChunkingResult:
    """Payload returned by the chunk builder."""

    chunks: List[ChunkCandidate]
    histogram: Dict[str, int]
    total_tokens: int


class TokenCounter:
    """Utility that converts text to tokens using tiktoken when available."""

    def __init__(self, *, encoding_name: str = "cl100k_base", use_tiktoken: bool = True) -> None:
        self._encoding_name = encoding_name
        self._encoder = None
        if use_tiktoken:
            try:  # pragma: no cover - tiktoken availability depends on environment
                import tiktoken

                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:  # pragma: no cover - fallback path covered in tests
                self._encoder = None
        self._fallback = self._encoder is None

    def encode(self, text: str) -> List[Any]:
        if self._fallback:
            return self._fallback_encode(text)
        return list(self._encoder.encode(text, disallowed_special=()))  # type: ignore[operator]

    def decode(self, tokens: Sequence[Any]) -> str:
        if self._fallback:
            return self._fallback_decode(tokens)
        return self._encoder.decode(tokens)  # type: ignore[call-arg]

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @staticmethod
    def _fallback_encode(text: str) -> List[str]:
        if not text:
            return []
        prepared = text.replace("\n", " \n ")
        return prepared.split()

    @staticmethod
    def _fallback_decode(tokens: Sequence[Any]) -> str:
        if not tokens:
            return ""
        joined = " ".join(str(token) for token in tokens)
        return joined.replace(" \n ", "\n").strip()


class ChunkBuilder:
    """Assembles deterministic text chunks bounded by token limits."""

    def __init__(
        self,
        *,
        config: ChunkConfig | None = None,
        tokenizer: SentenceTokenizer | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or ChunkConfig()
        self._tokenizer = tokenizer or SentenceTokenizer()
        self._counter = token_counter or TokenCounter(use_tiktoken=self._config.use_tiktoken)

    def build(self, *, document_text: str) -> ChunkingResult:
        if not document_text or not document_text.strip():
            return ChunkingResult(chunks=[], histogram={}, total_tokens=0)

        segment_queue = list(self._prepare_segments(document_text))
        if not segment_queue:
            return ChunkingResult(chunks=[], histogram={}, total_tokens=0)

        chunks: List[ChunkCandidate] = []
        token_counts: List[int] = []
        current_tokens: List[Any] = []
        carried_overlap = False

        for segment_text, segment_tokens in segment_queue:
            if current_tokens:
                combined = len(current_tokens) + len(segment_tokens)
                if combined > self._config.hard_max_tokens:
                    excess = combined - self._config.hard_max_tokens
                    if excess >= len(current_tokens):
                        current_tokens.clear()
                    else:
                        del current_tokens[:excess]
                        carried_overlap = len(current_tokens) > 0
            current_tokens.extend(segment_tokens)
            carried_overlap = False
            if len(current_tokens) >= self._config.target_tokens:
                self._finalize_chunk(chunks, token_counts, current_tokens)
                carried_overlap = self._config.overlap_tokens > 0

        if current_tokens and (not carried_overlap or len(current_tokens) > self._config.overlap_tokens):
            self._finalize_chunk(chunks, token_counts, current_tokens)

        histogram = self._build_histogram(token_counts)
        total_tokens = sum(token_counts)
        return ChunkingResult(chunks=chunks, histogram=histogram, total_tokens=total_tokens)

    def _prepare_segments(self, text: str) -> Iterable[Tuple[str, Tuple[Any, ...]]]:
        for segment in self._tokenizer.tokenize(text):
            raw_tokens = self._counter.encode(segment.text)
            if not raw_tokens:
                continue
            if len(raw_tokens) <= self._config.hard_max_tokens:
                yield (segment.text, tuple(raw_tokens))
                continue
            yield from self._split_segment_tokens(raw_tokens)

    def _split_segment_tokens(self, tokens: Sequence[Any]) -> Iterable[Tuple[str, Tuple[Any, ...]]]:
        max_tokens = self._config.hard_max_tokens
        for index in range(0, len(tokens), max_tokens):
            token_slice = tokens[index : index + max_tokens]
            text_slice = self._counter.decode(token_slice)
            yield (text_slice, tuple(token_slice))

    def _finalize_chunk(
        self,
        chunks: List[ChunkCandidate],
        token_counts: List[int],
        current_tokens: List[Any],
    ) -> None:
        if not current_tokens:
            return
        token_count = len(current_tokens)
        if token_count > self._config.hard_max_tokens:
            raise ValueError("Chunk exceeds hard token limit")
        chunk_text = self._counter.decode(current_tokens).strip()
        sequence_index = len(chunks)
        chunk_tokens = tuple(current_tokens)
        chunks.append(
            ChunkCandidate(
                sequence_index=sequence_index,
                text=chunk_text,
                token_count=token_count,
                tokens=chunk_tokens,
            )
        )
        token_counts.append(token_count)
        if self._config.overlap_tokens > 0 and token_count > 0:
            overlap_size = min(self._config.overlap_tokens, token_count)
            tail_tokens = list(chunk_tokens[-overlap_size:])
            current_tokens.clear()
            current_tokens.extend(tail_tokens)
        else:
            current_tokens.clear()

    def _build_histogram(self, token_counts: Sequence[int]) -> Dict[str, int]:
        if not token_counts:
            return {}
        bucket_size = self._config.histogram_bucket_size
        histogram: Dict[str, int] = defaultdict(int)
        for count in token_counts:
            bucket_index = 0 if count == 0 else (count - 1) // bucket_size
            start = bucket_index * bucket_size
            end = start + bucket_size
            label = f"{start}-{end}"
            histogram[label] += 1
        ordered = dict(sorted(histogram.items(), key=lambda item: int(item[0].split("-")[0])))
        return ordered