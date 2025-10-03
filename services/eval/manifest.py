from __future__ import annotations

import hashlib
import json
from threading import Lock
from typing import Dict, List

from services.eval.context_capture import CapturedEvaluationItem


class EvaluationManifestBuilder:
    """Builds and validates an integrity manifest for evaluation items (TASK-033c)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._items: List[Dict[str, object]] = []
        self._total_bytes = 0
        self._success_count = 0
        self._failure_count = 0
        self._closed = False
        self._digest = hashlib.sha256()

    def record(self, item: CapturedEvaluationItem, line_bytes: bytes) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Manifest builder is closed")
            entry = {
                "sample_id": item.sample_id,
                "run_id": item.run_id,
                "success": item.success,
                "error_code": item.error_code,
                "context_count": len(item.contexts),
                "bytes": len(line_bytes),
            }
            self._items.append(entry)
            self._total_bytes += len(line_bytes)
            if item.success:
                self._success_count += 1
            else:
                self._failure_count += 1
            self._digest.update(line_bytes)

    def finalize(self, *, file_size: int) -> Dict[str, object]:
        with self._lock:
            self._closed = True
            if file_size != self._total_bytes:
                raise RuntimeError(
                    f"Manifest byte mismatch: tracked={self._total_bytes} actual={file_size}"
                )
            return {
                "item_count": len(self._items),
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "total_bytes": self._total_bytes,
                "checksum": {
                    "algorithm": "sha256",
                    "value": self._digest.hexdigest(),
                },
                "items": list(self._items),
            }

    def to_json(self, *, file_size: int) -> str:
        manifest = self.finalize(file_size=file_size)
        return json.dumps(manifest, ensure_ascii=False, indent=2)

__all__ = ["EvaluationManifestBuilder"]
