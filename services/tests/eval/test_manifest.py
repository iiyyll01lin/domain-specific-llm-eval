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
