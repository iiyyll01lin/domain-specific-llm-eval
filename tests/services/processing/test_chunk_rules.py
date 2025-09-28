from __future__ import annotations

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.processing.stages import ChunkBuilder, ChunkConfig, TokenCounter  # noqa: E402


def _builder_for_testing(**config_overrides) -> ChunkBuilder:
    config = ChunkConfig(use_tiktoken=False, **config_overrides)
    counter = TokenCounter(use_tiktoken=False)
    return ChunkBuilder(config=config, token_counter=counter)


def test_chunk_builder_respects_limits_and_histogram():
    builder = _builder_for_testing(target_tokens=5, hard_max_tokens=8, overlap_tokens=2, histogram_bucket_size=2)
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

    result = builder.build(document_text=text)

    assert len(result.chunks) >= 2
    assert max(chunk.token_count for chunk in result.chunks) <= 8
    assert result.total_tokens == sum(chunk.token_count for chunk in result.chunks)
    assert sum(result.histogram.values()) == len(result.chunks)

    repeat = builder.build(document_text=text)
    assert [chunk.tokens for chunk in repeat.chunks] == [chunk.tokens for chunk in result.chunks]


def test_chunk_builder_applies_overlap_tokens():
    builder = _builder_for_testing(target_tokens=6, hard_max_tokens=8, overlap_tokens=3)
    text_segments: List[str] = []
    for idx in range(1, 7):
        segment_tokens = " ".join(f"token{idx}_{i}" for i in range(6))
        text_segments.append(f"{segment_tokens}.")
    text = " ".join(text_segments)

    result = builder.build(document_text=text)

    assert len(result.chunks) >= 2
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    matched = False
    max_overlap = builder._config.overlap_tokens
    for size in range(max_overlap, 0, -1):
        actual = min(size, len(first_chunk.tokens), len(second_chunk.tokens))
        if actual == 0:
            continue
        if first_chunk.tokens[-actual:] == second_chunk.tokens[:actual]:
            matched = True
            break
    assert matched


def test_chunk_builder_splits_long_segment():
    builder = _builder_for_testing(target_tokens=10, hard_max_tokens=12, overlap_tokens=0)
    words = " ".join(f"word{i}" for i in range(25))

    result = builder.build(document_text=words)

    assert len(result.chunks) == 3
    assert all(chunk.token_count <= builder._config.hard_max_tokens for chunk in result.chunks)
    assert result.total_tokens == 25
    assert result.chunks[0].token_count >= builder._config.target_tokens
    assert result.chunks[1].token_count >= builder._config.target_tokens


def test_chunk_builder_handles_empty_text():
    builder = _builder_for_testing()

    result = builder.build(document_text="   ")

    assert result.chunks == []
    assert result.histogram == {}
    assert result.total_tokens == 0


def test_chunk_builder_rejects_invalid_config():
    with pytest.raises(ValueError):
        ChunkConfig(target_tokens=0)
    with pytest.raises(ValueError):
        ChunkConfig(target_tokens=10, hard_max_tokens=5)
    with pytest.raises(ValueError):
        ChunkConfig(target_tokens=10, hard_max_tokens=10, overlap_tokens=-1)
    with pytest.raises(ValueError):
        ChunkConfig(target_tokens=10, hard_max_tokens=10, histogram_bucket_size=0)

