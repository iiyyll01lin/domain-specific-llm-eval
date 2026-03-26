from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_dev_parity.py"
_SPEC = importlib.util.spec_from_file_location("root_validate_dev_parity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validate_dev_parity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validate_dev_parity
_SPEC.loader.exec_module(validate_dev_parity)


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


def test_compare_snapshots_marks_python_warn_severity() -> None:
    current = {
        "python": {"expected": "3.11", "actual": "3.10"},
        "dependency_files": {},
        "extensions": {},
    }
    expected = {
        "python": {"expected": "3.11", "actual": "3.11"},
        "dependency_files": {},
        "extensions": {},
    }

    drifts = validate_dev_parity.compare_snapshots(
        current,
        expected,
        python_drift_severity="warn",
    )

    assert len(drifts) == 1
    assert drifts[0].severity == "warn"


def test_build_markdown_report_contains_whitelisted_drift() -> None:
    report = {
        "python": {"expected": "3.11", "actual": "3.10"},
        "python_drift_severity": "warn",
        "drifts": [
            {
                "category": "python",
                "name": "python_version",
                "expected": "3.11",
                "actual": "3.10",
                "severity": "warn",
                "whitelisted": True,
            }
        ],
        "failures": [],
        "warnings": ["python_version"],
    }

    markdown = validate_dev_parity.build_markdown_report(report)

    assert "python_version" in markdown
    assert "True" in markdown
    assert "warn" in markdown


def test_load_whitelist_reads_json(tmp_path: Path) -> None:
    whitelist_path = tmp_path / "whitelist.json"
    whitelist_path.write_text(
        json.dumps({"extensions": ["sample_metric.py"], "python": ["python_version"]}),
        encoding="utf-8",
    )

    whitelist = validate_dev_parity.load_whitelist(whitelist_path)

    assert "sample_metric.py" in whitelist["extensions"]
    assert "python_version" in whitelist["python"]


def test_evaluate_parity_warns_for_local_python_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_dev_parity,
        "build_snapshot",
        lambda: {
            "python": {"expected": "3.11", "actual": "3.10"},
            "dependency_files": {},
            "extensions": {},
        },
    )

    class Args:
        strict = False
        skip_installed_packages = True
        snapshot_json = None
        whitelist = None
        python_drift_severity = "warn"

    report = validate_dev_parity.evaluate_parity(Args())

    assert report["failures"] == []
    assert report["warnings"] == ["python_version"]
    assert report["python_drift_severity"] == "warn"