from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Callable, Iterable, Optional

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.manifest import EvaluationManifestBuilder
from services.eval.persistence_metrics import PersistenceMetricsRecorder
from services.eval.rag_interface import RetrievedContext


def _serialize_contexts(contexts: Iterable[RetrievedContext]) -> list[dict[str, object | None]]:
    return [context.as_dict() for context in contexts]


def _serialize_item(item: CapturedEvaluationItem) -> dict[str, object | None]:
    return {
        "run_id": item.run_id,
        "sample_id": item.sample_id,
        "question": item.question,
        "answer": item.answer,
        "success": item.success,
        "error_code": item.error_code,
        "metadata": dict(item.metadata),
        "raw": dict(item.raw),
        "contexts": _serialize_contexts(item.contexts),
    }


class EvaluationItemStreamWriter:
    """Incrementally writes evaluation items to a JSON Lines file (TASK-033a).

    The writer buffers serialised records in-memory and flushes them to disk
    whenever the configured ``flush_interval_seconds`` elapses or when
    :meth:`flush` / :meth:`close` is invoked explicitly. The implementation is
    intentionally lightweight; higher-level orchestration performs
    backpressure/queue handling (TASK-033b) and manifest generation (TASK-033c).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        flush_interval_seconds: float = 1.0,
        time_provider: Callable[[], float] = monotonic,
        encoding: str = "utf-8",
        manifest_builder: Optional[EvaluationManifestBuilder] = None,
        manifest_path: Optional[str | os.PathLike[str]] = None,
        metrics: Optional[PersistenceMetricsRecorder] = None,
    ) -> None:
        if flush_interval_seconds < 0:
            raise ValueError("flush_interval_seconds must be >= 0")
        self._path = Path(path)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_seconds
        self._time_provider = time_provider
        self._encoding = encoding
        self._buffer: list[tuple[str, bytes]] = []
        self._buffer_size_bytes = 0
        self._manifest = manifest_builder or EvaluationManifestBuilder()
        self._manifest_path = Path(manifest_path) if manifest_path else self._path.with_suffix(
            self._path.suffix + ".manifest.json"
        )
        if not self._manifest_path.parent.exists():
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics = metrics
        self._total_items = 0
        self._last_flush = self._time_provider()
        self._lock = Lock()
        self._file = self._path.open("w", encoding=self._encoding)
        self._closed = False
        self._manifest_data: Optional[dict[str, object]] = None

    @property
    def path(self) -> Path:
        return self._path

    def write(self, item: CapturedEvaluationItem) -> None:
        payload = _serialize_item(item)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        line_bytes = serialized.encode(self._encoding)
        with self._lock:
            self._buffer.append((serialized, line_bytes))
            self._buffer_size_bytes += len(line_bytes)
            self._total_items += 1
            if self._manifest is not None:
                self._manifest.record(item=item, line_bytes=line_bytes)
            if self._metrics is not None:
                self._metrics.record_item_written(bytes_written=len(line_bytes))
            now = self._time_provider()
            if self._flush_interval == 0 or now - self._last_flush >= self._flush_interval:
                self._flush_locked(now)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked(self._time_provider())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked(self._time_provider())
            self._file.close()
            file_size = self._path.stat().st_size if self._path.exists() else 0
            self._manifest_data = self._manifest.finalize(file_size=file_size)
            manifest_payload = json.dumps(self._manifest_data, ensure_ascii=False, indent=2)
            self._manifest_path.write_text(manifest_payload, encoding=self._encoding)
            self._closed = True

    def _flush_locked(self, timestamp: Optional[float] = None) -> None:
        if not self._buffer:
            return
        start = self._time_provider()
        lines, byte_chunks = zip(*self._buffer)
        self._file.writelines(lines)
        self._file.flush()
        duration = max(self._time_provider() - start, 0.0)
        bytes_written = sum(len(chunk) for chunk in byte_chunks)
        if self._metrics is not None:
            self._metrics.record_flush(
                item_count=len(byte_chunks),
                bytes_written=bytes_written,
                latency_seconds=duration,
            )
        self._buffer.clear()
        self._buffer_size_bytes = 0
        self._last_flush = self._time_provider() if timestamp is None else timestamp

    def __enter__(self) -> "EvaluationItemStreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def manifest_data(self) -> Optional[dict[str, object]]:
        return self._manifest_data


__all__ = ["EvaluationItemStreamWriter"]
