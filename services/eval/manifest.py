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


# ---------------------------------------------------------------------------
# Run-level artifact manifest  (TASK-083)
# ---------------------------------------------------------------------------

import os
import datetime
from pathlib import Path


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_run_manifest(
    run_id: str,
    artifacts: List[str | Path],
    *,
    output_path: str | Path | None = None,
) -> Dict[str, object]:
    """Generate a run-level artifact manifest with sha256 checksums.

    Parameters
    ----------
    run_id:
        Unique identifier for the pipeline run.
    artifacts:
        Sequence of file paths that belong to this run.  Non-existent paths
        are recorded with ``exists: false`` and no checksum.
    output_path:
        If provided, the manifest JSON is written to this path.

    Returns
    -------
    dict
        The manifest document (mirrors the written JSON).
    """
    artifact_records: List[Dict[str, object]] = []
    for raw_path in artifacts:
        p = Path(raw_path)
        if not p.exists():
            artifact_records.append({
                "path": str(p),
                "exists": False,
            })
        else:
            artifact_records.append({
                "path": str(p),
                "exists": True,
                "size_bytes": p.stat().st_size,
                "checksum": {
                    "algorithm": "sha256",
                    "value": _sha256_file(p),
                },
            })

    missing = [a for a in artifact_records if not a["exists"]]
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "artifact_count": len(artifact_records),
        "missing_count": len(missing),
        "artifacts": artifact_records,
    }

    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


__all__ = ["EvaluationManifestBuilder", "generate_run_manifest"]

