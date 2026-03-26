"""Tests for scripts/check_bundle_size.py (TASK-082)."""
import gzip
import json
import pathlib
import subprocess
import sys
import tempfile

import pytest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_bundle_size.py"


def _write_js(directory: pathlib.Path, name: str, size_bytes: int) -> pathlib.Path:
    """Write a JS file of exactly *size_bytes* to *directory*."""
    path = directory / name
    path.write_bytes(b"x" * size_bytes)
    return path


class TestCheckBundleSize:
    def test_pass_when_chunk_within_budget(self, tmp_path):
        """A KgPanel chunk under 300 KB gz should return exit code 0."""
        # Write a small file (will compress well below 300 KB)
        _write_js(tmp_path, "KgPanel-abc123.js", 1024)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path), "--budget-kb", "300"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_fail_when_chunk_exceeds_budget(self, tmp_path):
        """A KgPanel chunk over budget should return exit code 1."""
        _write_js(tmp_path, "KgPanel-big.js", 4096)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path), "--budget-kb", "0"],  # 0 KB budget → always fails
            capture_output=True, text=True,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "OVER" in result.stdout or "FAIL" in result.stdout

    def test_non_kg_chunks_not_subject_to_budget(self, tmp_path):
        """Non-KG chunks should not cause failure even if large."""
        _write_js(tmp_path, "vendor-huge.js", 1024 * 1024)  # 1 MB, not KG
        _write_js(tmp_path, "KgPanel-small.js", 1024)        # tiny KG chunk
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path), "--budget-kb", "300"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_report_written_to_json(self, tmp_path):
        """--report flag should produce a valid JSON report file."""
        _write_js(tmp_path, "KgPanel-xyz.js", 10 * 1024)
        report_path = tmp_path / "report.json"
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path),
             "--budget-kb", "300",
             "--report", str(report_path)],
            capture_output=True, text=True, check=True,
        )
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "budget_kb" in data
        assert "chunks" in data
        assert isinstance(data["chunks"], list)

    def test_report_contains_kg_chunk_entry(self, tmp_path):
        """Report JSON should include the KG chunk with gz_kb field."""
        _write_js(tmp_path, "KgPanel-test.js", 5 * 1024)
        report_path = tmp_path / "r.json"
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path),
             "--budget-kb", "300",
             "--report", str(report_path)],
            capture_output=True, text=True,
        )
        data = json.loads(report_path.read_text())
        kg_chunks = [c for c in data["chunks"] if c["is_kg_chunk"]]
        assert len(kg_chunks) >= 1
        assert "gz_kb" in kg_chunks[0]
        assert kg_chunks[0]["raw_kb"] > 0  # raw size must be positive

    def test_missing_build_dir_exits_nonzero(self, tmp_path):
        """Missing build directory should exit with code 1."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path / "nonexistent")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_empty_dir_exits_nonzero(self, tmp_path):
        """A build dir with no JS files should exit 1."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_custom_pattern_respected(self, tmp_path):
        """Custom --pattern should be matched case-insensitively."""
        _write_js(tmp_path, "MyCustomChunk-abc.js", 5 * 1024)
        # With budget of 0 KB any file will fail
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--build-dir", str(tmp_path),
             "--budget-kb", "0",
             "--pattern", "mycustom"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
