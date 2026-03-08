from __future__ import annotations

import json
from pathlib import Path

from scripts import validate_dev_parity


def test_compare_snapshots_detects_dependency_drift() -> None:
    current = {
        "python": {"expected": "3.11", "actual": "3.11"},
        "dependency_files": {"requirements.txt": "new-hash"},
        "extensions": {},
    }
    expected = {
        "python": {"expected": "3.11", "actual": "3.11"},
        "dependency_files": {"requirements.txt": "old-hash"},
        "extensions": {},
    }

    drifts = validate_dev_parity.compare_snapshots(current, expected)

    assert len(drifts) == 1
    assert drifts[0].category == "dependency_files"
    assert drifts[0].name == "requirements.txt"


def test_compare_snapshots_respects_whitelist() -> None:
    current = {
        "python": {"expected": "3.11", "actual": "3.10"},
        "dependency_files": {},
        "extensions": {"sample.py": "new"},
    }
    expected = {
        "python": {"expected": "3.11", "actual": "3.11"},
        "dependency_files": {},
        "extensions": {"sample.py": "old"},
    }

    drifts = validate_dev_parity.compare_snapshots(
        current,
        expected,
        {"python": {"python_version"}, "dependency_files": set(), "extensions": {"sample.py"}, "packages": set()},
    )

    assert len(drifts) == 2
    assert all(drift.whitelisted for drift in drifts)


def test_build_markdown_report_contains_whitelisted_drift() -> None:
    report = {
        "python": {"expected": "3.11", "actual": "3.10"},
        "drifts": [
            {
                "category": "python",
                "name": "python_version",
                "expected": "3.11",
                "actual": "3.10",
                "whitelisted": True,
            }
        ],
        "failures": [],
        "warnings": ["python_version"],
    }

    markdown = validate_dev_parity.build_markdown_report(report)

    assert "python_version" in markdown
    assert "True" in markdown


def test_load_whitelist_reads_json(tmp_path: Path) -> None:
    whitelist_path = tmp_path / "whitelist.json"
    whitelist_path.write_text(
        json.dumps({"extensions": ["sample_metric.py"], "python": ["python_version"]}),
        encoding="utf-8",
    )

    whitelist = validate_dev_parity.load_whitelist(whitelist_path)

    assert "sample_metric.py" in whitelist["extensions"]
    assert "python_version" in whitelist["python"]