from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def build_benchmark_comment(summary: Dict[str, Any]) -> str:
    checks: Iterable[Dict[str, Any]] = summary.get("checks", [])
    rows: List[str] = []
    for check in checks:
        status = "✅" if check.get("ok", False) else "❌"
        rows.append(
            f"| {check.get('name', 'unknown')} | {status} | {check.get('details', '')} |"
        )

    body = [
        "## Eval Pipeline Benchmark Summary",
        "",
        f"Run ID: `{summary.get('run_id', 'unknown')}`",
        "",
        "| Check | Status | Details |",
        "|-------|--------|---------|",
        *rows,
    ]

    if summary.get("artifacts"):
        body.extend(["", "Artifacts:"])
        for artifact in summary["artifacts"]:
            body.append(f"- {artifact}")

    return "\n".join(body)


def write_benchmark_comment(report_path: Path, output_path: Path) -> None:
    with open(report_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    output_path.write_text(build_benchmark_comment(summary), encoding="utf-8")