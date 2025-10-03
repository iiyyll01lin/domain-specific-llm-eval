from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping


class KPIWriter:
    """Atomically writes KPI aggregation results to disk (TASK-034c)."""

    def __init__(self, path: str | os.PathLike[str], *, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._encoding = encoding

    @property
    def path(self) -> Path:
        return self._path

    def write(self, payload: Mapping[str, object]) -> Path:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        directory = self._path.parent
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding=self._encoding) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self._path)
        return self._path


__all__ = ["KPIWriter"]
