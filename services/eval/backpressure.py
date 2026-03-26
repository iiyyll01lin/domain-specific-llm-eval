from __future__ import annotations

import logging
from collections import deque
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Deque, Optional

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.stream_writer import EvaluationItemStreamWriter

logger = logging.getLogger(__name__)


class EvaluationItemQueueWorker:
    """Background worker consuming evaluation items with backpressure (TASK-033b)."""

    def __init__(
        self,
        stream_writer: EvaluationItemStreamWriter,
        *,
        max_queue_size: int = 1024,
        drop_oldest: bool = True,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        self._stream_writer = stream_writer
        self._max_queue_size = max_queue_size
        self._drop_oldest = drop_oldest
        self._queue: Deque[CapturedEvaluationItem] = deque()
        self._lock = Lock()
        self._has_items = Condition(self._lock)
        self._stopping = False
        self._dropped_total = 0
        self._thread = Thread(target=self._drain_loop, name="evaluation-item-writer", daemon=True)
        self._thread.start()

    @property
    def dropped_total(self) -> int:
        return self._dropped_total

    def submit(self, item: CapturedEvaluationItem) -> None:
        with self._lock:
            if self._stopping:
                raise RuntimeError("worker has been stopped")
            if len(self._queue) >= self._max_queue_size:
                dropped = self._queue.popleft() if self._drop_oldest else self._queue.pop()
                self._dropped_total += 1
                logger.warning(
                    "eval.persistence.backpressure",
                    extra={
                        "context": {
                            "run_id": dropped.run_id,
                            "sample_id": dropped.sample_id,
                            "queue_size": len(self._queue),
                            "max_queue_size": self._max_queue_size,
                            "dropped_total": self._dropped_total,
                        }
                    },
                )
            self._queue.append(item)
            self._has_items.notify()

    def stop(self, *, drain: bool = True) -> None:
        with self._lock:
            self._stopping = True
            self._has_items.notify_all()
        self._thread.join()
        if drain:
            self._stream_writer.flush()
        self._stream_writer.close()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait until the queue is empty. Returns ``True`` if drained."""

        deadline = None if timeout is None else monotonic() + timeout
        with self._lock:
            while self._queue:
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return False
                    self._has_items.wait(timeout=remaining)
                else:
                    self._has_items.wait()
            return True

    def _drain_loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stopping:
                    self._has_items.wait()
                if not self._queue:
                    if self._stopping:
                        break
                    continue
                item = self._queue.popleft()
                if not self._queue:
                    self._has_items.notify_all()
            try:
                self._stream_writer.write(item)
            except Exception:  # pragma: no cover - defensively log and continue
                logger.exception(
                    "eval.persistence.write_error",
                    extra={
                        "context": {
                            "run_id": item.run_id,
                            "sample_id": item.sample_id,
                        }
                    },
                )

    def __enter__(self) -> "EvaluationItemQueueWorker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = ["EvaluationItemQueueWorker"]
