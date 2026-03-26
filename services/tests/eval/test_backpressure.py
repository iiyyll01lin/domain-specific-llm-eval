from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Event

import pytest

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.rag_interface import RetrievedContext
from services.eval.stream_writer import EvaluationItemStreamWriter
from services.eval.backpressure import EvaluationItemQueueWorker


def _make_item(sample_id: str) -> CapturedEvaluationItem:
    return CapturedEvaluationItem(
        run_id="run-xyz",
        sample_id=sample_id,
        question="Dummy question",
        answer="Dummy answer",
        contexts=(
            RetrievedContext(
                text="Some context",
                document_id="doc",
                score=0.5,
                metadata={}
            ),
        ),
        success=True,
        metadata={},
        raw={},
    )


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_queue_worker_drains_items(tmp_path: Path) -> None:
    target = tmp_path / "evaluation_items.jsonl"
    stream_writer = EvaluationItemStreamWriter(target, flush_interval_seconds=0.0)
    worker = EvaluationItemQueueWorker(stream_writer, max_queue_size=8)

    try:
        worker.submit(_make_item("sample-001"))
        worker.submit(_make_item("sample-002"))

        assert _wait_for(lambda: target.exists() and target.stat().st_size > 0)
    finally:
        worker.stop()

    contents = target.read_text(encoding="utf-8").strip().splitlines()
    samples = [json.loads(line)["sample_id"] for line in contents]
    assert samples == ["sample-001", "sample-002"]

    manifest = json.loads(stream_writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["item_count"] == 2


def test_queue_worker_logs_backpressure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class SlowWriter:
        def __init__(self) -> None:
            self._event = Event()
            self.written: list[str] = []

        def write(self, item: CapturedEvaluationItem) -> None:
            self._event.wait()
            self.written.append(item.sample_id)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def release(self) -> None:
            self._event.set()

    slow_writer = SlowWriter()
    worker = EvaluationItemQueueWorker(slow_writer, max_queue_size=2)

    caplog.set_level(logging.WARNING)

    try:
        for index in range(4):
            worker.submit(_make_item(f"sample-{index}"))

        assert _wait_for(lambda: worker.dropped_total >= 1)
        assert any(record.getMessage() == "eval.persistence.backpressure" for record in caplog.records)
        drop_record = next(record for record in caplog.records if record.getMessage() == "eval.persistence.backpressure")
        assert drop_record.context["dropped_total"] >= 1
    finally:
        slow_writer.release()
        worker.stop()


def test_submit_after_stop_raises(tmp_path: Path) -> None:
    stream_writer = EvaluationItemStreamWriter(tmp_path / "items.jsonl", flush_interval_seconds=0.0)
    worker = EvaluationItemQueueWorker(stream_writer, max_queue_size=1)

    worker.stop()

    with pytest.raises(RuntimeError):
        worker.submit(_make_item("sample-stop"))
