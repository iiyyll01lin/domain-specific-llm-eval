from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.rag_interface import RetrievedContext
from services.eval.stream_writer import EvaluationItemStreamWriter


class _FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def advance(self, seconds: float) -> None:
        self._value += seconds

    def monotonic(self) -> float:
        return self._value


def _make_item(sample_id: str = "sample-001") -> CapturedEvaluationItem:
    return CapturedEvaluationItem(
        run_id="run-123",
        sample_id=sample_id,
        question="What is the capital of Taiwan?",
        answer="Taipei",
        contexts=(
            RetrievedContext(
                text="Taipei is the capital city of Taiwan.",
                document_id="doc-1",
                score=0.98,
                metadata={"source": "wikipedia"},
            ),
        ),
        success=True,
        metadata={"attempts": 1},
        raw={"latency_ms": 123.4},
    )


def test_stream_writer_persists_items(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "run-123" / "evaluation_items.jsonl"
    item = _make_item()

    with EvaluationItemStreamWriter(target, flush_interval_seconds=0.0) as writer:
        writer.write(item)

    assert target.exists()
    contents = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["run_id"] == item.run_id
    assert payload["sample_id"] == item.sample_id
    assert payload["question"] == item.question
    assert payload["answer"] == item.answer
    assert payload["success"] is True
    assert payload["contexts"][0]["document_id"] == "doc-1"
    assert payload["metadata"]["attempts"] == 1
    assert payload["raw"]["latency_ms"] == pytest.approx(123.4)

    manifest_path = writer.manifest_path
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["item_count"] == 1
    assert manifest_payload["total_bytes"] == len(target.read_bytes())


def test_stream_writer_flushes_on_interval(tmp_path: Path) -> None:
    target = tmp_path / "evaluation_items.jsonl"
    clock = _FakeClock()
    writer = EvaluationItemStreamWriter(target, flush_interval_seconds=5.0, time_provider=clock.monotonic)

    writer.write(_make_item("sample-001"))
    assert target.read_text(encoding="utf-8") == ""

    clock.advance(6.0)
    writer.write(_make_item("sample-002"))

    writer.close()

    contents = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2
    samples = [json.loads(line)["sample_id"] for line in contents]
    assert samples == ["sample-001", "sample-002"]

    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["item_count"] == 2
    assert manifest["success_count"] == 2


def test_stream_writer_close_flushes_remaining_buffer(tmp_path: Path) -> None:
    target = tmp_path / "evaluation_items.jsonl"
    writer = EvaluationItemStreamWriter(target, flush_interval_seconds=30.0)

    writer.write(_make_item("sample-101"))
    writer.close()

    contents = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    assert json.loads(contents[0])["sample_id"] == "sample-101"
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure_count"] == 0
    assert manifest["checksum"]["algorithm"] == "sha256"
