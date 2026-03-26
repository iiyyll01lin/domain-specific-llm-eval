from __future__ import annotations

from services.eval.context_capture import CapturedEvaluationItem
from services.eval.manifest import EvaluationManifestBuilder
from services.eval.rag_interface import RetrievedContext


def _make_item(sample_id: str = "sample-001", success: bool = True) -> CapturedEvaluationItem:
    return CapturedEvaluationItem(
        run_id="run-abc",
        sample_id=sample_id,
        question="Q?",
        answer="A!",
        contexts=(RetrievedContext(text="ctx"),),
        success=success,
        metadata={},
        raw={},
    )


def test_manifest_builder_accumulates_items() -> None:
    builder = EvaluationManifestBuilder()
    for i in range(3):
        builder.record(_make_item(f"sample-{i}"), line_bytes=b"abcde")
    manifest = builder.finalize(file_size=15)
    assert manifest["item_count"] == 3
    assert manifest["success_count"] == 3
    assert manifest["failure_count"] == 0
    assert manifest["total_bytes"] == 15
    assert len(manifest["items"]) == 3
    assert manifest["checksum"]["value"]


def test_manifest_builder_detects_mismatch() -> None:
    builder = EvaluationManifestBuilder()
    builder.record(_make_item(success=False), line_bytes=b"abc")
    try:
        builder.finalize(file_size=10)
        assert False, "Expected mismatch error"
    except RuntimeError as exc:
        assert "Manifest byte mismatch" in str(exc)


def test_manifest_builder_blocks_after_finalize() -> None:
    builder = EvaluationManifestBuilder()
    builder.record(_make_item(), line_bytes=b"abc")
    builder.finalize(file_size=3)
    try:
        builder.record(_make_item("sample-x"), line_bytes=b"def")
        assert False
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# generate_run_manifest tests (TASK-083)
# ---------------------------------------------------------------------------

import hashlib
import json
import tempfile
from pathlib import Path

from services.eval.manifest import generate_run_manifest


def test_run_manifest_checksum_correct(tmp_path: Path) -> None:
    """SHA-256 in manifest matches independently computed digest."""
    artifact = tmp_path / "evaluation_items.json"
    artifact.write_bytes(b'{"result": "ok"}')

    manifest = generate_run_manifest("run-xyz", [artifact])
    assert manifest["artifact_count"] == 1
    assert manifest["missing_count"] == 0

    record = manifest["artifacts"][0]  # type: ignore[index]
    assert record["exists"] is True
    expected = hashlib.sha256(b'{"result": "ok"}').hexdigest()
    assert record["checksum"]["value"] == expected


def test_run_manifest_missing_artifact(tmp_path: Path) -> None:
    """Missing artifact is recorded with exists=False and no checksum."""
    missing = tmp_path / "does_not_exist.json"
    manifest = generate_run_manifest("run-missing", [missing])

    assert manifest["missing_count"] == 1
    record = manifest["artifacts"][0]  # type: ignore[index]
    assert record["exists"] is False
    assert "checksum" not in record


def test_run_manifest_writes_file(tmp_path: Path) -> None:
    """generate_run_manifest writes valid JSON to output_path."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"total": 5}', encoding="utf-8")
    output = tmp_path / "manifest.json"

    generate_run_manifest("run-write", [artifact], output_path=output)

    assert output.exists()
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["run_id"] == "run-write"
    assert parsed["schema_version"] == 1


def test_run_manifest_multiple_artifacts(tmp_path: Path) -> None:
    """Manifest records all artifacts including mixed present/missing."""
    present = tmp_path / "a.txt"
    present.write_bytes(b"hello")
    absent = tmp_path / "b.txt"

    manifest = generate_run_manifest("run-multi", [present, absent])
    assert manifest["artifact_count"] == 2
    assert manifest["missing_count"] == 1


def test_run_manifest_schema_version() -> None:
    """schema_version field is present and equals 1."""
    manifest = generate_run_manifest("run-sv", [])
    assert manifest["schema_version"] == 1

