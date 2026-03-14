from __future__ import annotations

import json
from pathlib import Path

from src.utils.benchmark_comment import build_benchmark_comment, write_benchmark_comment


def test_build_benchmark_comment_renders_markdown_table() -> None:
    body = build_benchmark_comment(
        {
            "run_id": "ci-benchmark",
            "checks": [
                {"name": "domain-metrics", "ok": True, "details": "passed"},
                {"name": "report-regression", "ok": False, "details": "1 failure"},
            ],
            "artifacts": ["artifact-a.json"],
        }
    )

    assert "## Eval Pipeline Benchmark Summary" in body
    assert "| domain-metrics | ✅ | passed |" in body
    assert "artifact-a.json" in body


def test_write_benchmark_comment_writes_output_file(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark-summary.json"
    output_path = tmp_path / "benchmark-comment.md"
    report_path.write_text(
        json.dumps({"run_id": "demo", "checks": [], "artifacts": []}),
        encoding="utf-8",
    )

    write_benchmark_comment(report_path, output_path)

    assert output_path.exists()
    assert "Run ID: `demo`" in output_path.read_text(encoding="utf-8")
