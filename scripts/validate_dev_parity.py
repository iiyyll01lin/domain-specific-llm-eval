#!/usr/bin/env python3
"""Dev/CI environment parity validation.

Checks that the local development environment stays aligned with repository and
container expectations. The validator supports three layers:

1. Python version parity against the base image declared in the Dockerfile.
2. Installed package parity against pinned requirements.
3. Snapshot parity for dependency files and extensions fingerprints.

The script emits optional JSON and Markdown reports so it can serve as a local
pre-push tool or a CI governance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
EXTENSIONS_DIR = REPO_ROOT / "extensions"
LOCK_FILES = ["requirements.txt", "pyproject.toml", "poetry.lock"]

_COMMENT_RE = re.compile(r"#.*")
_VER_SPEC_RE = re.compile(r"[><=!~^]+\s*[\w.*]+")
_DOCKER_PYTHON_RE = re.compile(r"^FROM\s+python:(?P<version>\d+\.\d+)", re.MULTILINE)


@dataclass(frozen=True)
class Drift:
    category: str
    name: str
    expected: str
    actual: str
    severity: str
    whitelisted: bool = False


def default_python_drift_severity() -> str:
    ci_value = str(os.environ.get("CI", "")).strip().lower()
    return "error" if ci_value in {"1", "true", "yes", "on"} else "warn"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_requirements(requirements_file: pathlib.Path = REQUIREMENTS_FILE) -> List[Tuple[str, str]]:
    if not requirements_file.exists():
        return []

    results: List[Tuple[str, str]] = []
    for raw in requirements_file.read_text(encoding="utf-8").splitlines():
        line = _COMMENT_RE.sub("", raw).strip()
        if not line or line.startswith("-") or line.startswith("git+"):
            continue
        name_part = _VER_SPEC_RE.split(line)[0].strip()
        name_part = re.sub(r"\[.*?\]", "", name_part).strip()
        pinned = ""
        pin_match = re.search(r"==\s*([\w.*-]+)", line)
        if pin_match:
            pinned = pin_match.group(1)
        if name_part:
            results.append((name_part, pinned))
    return results


def _read_expected_python_version(dockerfile: pathlib.Path = DOCKERFILE) -> str | None:
    if not dockerfile.exists():
        return None
    match = _DOCKER_PYTHON_RE.search(dockerfile.read_text(encoding="utf-8"))
    if match is None:
        return None
    return match.group("version")


def _collect_dependency_hashes(repo_root: pathlib.Path = REPO_ROOT) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in LOCK_FILES:
        path = repo_root / relative
        if path.exists():
            hashes[relative] = _sha256(path)
    return hashes


def _collect_extension_hashes(extensions_dir: pathlib.Path = EXTENSIONS_DIR) -> Dict[str, str]:
    if not extensions_dir.exists():
        return {}
    hashes: Dict[str, str] = {}
    for path in sorted(extensions_dir.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(extensions_dir))] = _sha256(path)
    return hashes


def build_snapshot(
    *,
    repo_root: pathlib.Path = REPO_ROOT,
    dockerfile: pathlib.Path = DOCKERFILE,
    extensions_dir: pathlib.Path = EXTENSIONS_DIR,
) -> Dict[str, Any]:
    return {
        "python": {
            "expected": _read_expected_python_version(dockerfile),
            "actual": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "dependency_files": _collect_dependency_hashes(repo_root),
        "extensions": _collect_extension_hashes(extensions_dir),
    }


def load_whitelist(path: pathlib.Path | None) -> Dict[str, set[str]]:
    if path is None or not path.exists():
        return {"python": set(), "dependency_files": set(), "extensions": set(), "packages": set()}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "python": set(data.get("python", [])),
        "dependency_files": set(data.get("dependency_files", [])),
        "extensions": set(data.get("extensions", [])),
        "packages": set(data.get("packages", [])),
    }


def compare_snapshots(
    current: Dict[str, Any],
    expected: Dict[str, Any],
    whitelist: Dict[str, set[str]] | None = None,
    *,
    python_drift_severity: str = "error",
) -> List[Drift]:
    whitelist = whitelist or {"python": set(), "dependency_files": set(), "extensions": set(), "packages": set()}
    drifts: List[Drift] = []

    current_python = current.get("python", {}).get("actual") or "unknown"
    expected_python = expected.get("python", {}).get("expected") or expected.get("python", {}).get("actual")
    if expected_python and current_python != expected_python:
        drifts.append(
            Drift(
                category="python",
                name="python_version",
                expected=str(expected_python),
                actual=str(current_python),
                severity=python_drift_severity,
                whitelisted="python_version" in whitelist.get("python", set()),
            )
        )

    for category in ("dependency_files", "extensions"):
        current_items = current.get(category, {})
        expected_items = expected.get(category, {})
        for name in sorted(set(current_items) | set(expected_items)):
            current_value = current_items.get(name, "<missing>")
            expected_value = expected_items.get(name, "<missing>")
            if current_value != expected_value:
                drifts.append(
                    Drift(
                        category=category,
                        name=name,
                        expected=str(expected_value),
                        actual=str(current_value),
                        severity="error",
                        whitelisted=name in whitelist.get(category, set()),
                    )
                )
    return drifts


def check_packages(strict: bool, *, skip_installed_packages: bool = False) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
    failures: List[str] = []
    warnings: List[str] = []
    packages_report: Dict[str, Dict[str, str]] = {}

    if skip_installed_packages:
        return failures, warnings, packages_report

    packages = _parse_requirements()
    for pkg_name, pinned in packages:
        dist_name = pkg_name.replace("_", "-")
        try:
            installed_version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Package not installed: {pkg_name}")
            packages_report[pkg_name] = {"expected": pinned or "present", "actual": "<missing>", "status": "missing"}
            continue

        status = "ok"
        if pinned and installed_version != pinned:
            status = "mismatch"
            message = f"Version mismatch: {pkg_name} installed={installed_version} pinned={pinned}"
            if strict:
                failures.append(message)
            else:
                warnings.append(message)
        packages_report[pkg_name] = {
            "expected": pinned or "present",
            "actual": installed_version,
            "status": status,
        }
    return failures, warnings, packages_report


def build_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Dev/CI Parity Report",
        "",
        f"- Python expected: `{report['python']['expected']}`",
        f"- Python actual: `{report['python']['actual']}`",
        f"- Python drift severity: `{report['python_drift_severity']}`",
        f"- Drift count: `{len(report['drifts'])}`",
        f"- Failure count: `{len(report['failures'])}`",
        f"- Warning count: `{len(report['warnings'])}`",
        "",
        "## Drift Summary",
        "",
        "| Category | Name | Expected | Actual | Severity | Whitelisted |",
        "|----------|------|----------|--------|----------|-------------|",
    ]
    if report["drifts"]:
        for drift in report["drifts"]:
            lines.append(
                f"| {drift['category']} | {drift['name']} | {drift['expected']} | {drift['actual']} | {drift['severity']} | {drift['whitelisted']} |"
            )
    else:
        lines.append("| none | none | none | none | none | False |")
    return "\n".join(lines) + "\n"


def evaluate_parity(args: argparse.Namespace) -> Dict[str, Any]:
    current = build_snapshot()
    expected = current
    if args.snapshot_json:
        expected = json.loads(pathlib.Path(args.snapshot_json).read_text(encoding="utf-8"))

    whitelist = load_whitelist(pathlib.Path(args.whitelist) if args.whitelist else None)
    python_drift_severity = args.python_drift_severity or default_python_drift_severity()
    drifts = compare_snapshots(
        current,
        expected,
        whitelist,
        python_drift_severity=python_drift_severity,
    )
    package_failures, package_warnings, package_report = check_packages(
        args.strict,
        skip_installed_packages=args.skip_installed_packages,
    )

    failures = [drift.name for drift in drifts if not drift.whitelisted and drift.severity == "error"]
    failures.extend(package_failures)
    warnings = [drift.name for drift in drifts if drift.whitelisted or drift.severity == "warn"]
    warnings.extend(package_warnings)

    return {
        "python": current["python"],
        "python_drift_severity": python_drift_severity,
        "dependency_files": current["dependency_files"],
        "extensions": current["extensions"],
        "drifts": [drift.__dict__ for drift in drifts],
        "failures": failures,
        "warnings": warnings,
        "packages": package_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat version mismatches as failures.")
    parser.add_argument("--skip-installed-packages", action="store_true", help="Skip installed package checks for lightweight CI validation.")
    parser.add_argument("--snapshot-json", help="Compare against a previously recorded snapshot JSON file.")
    parser.add_argument("--write-snapshot", help="Write the current snapshot JSON to the given path and exit.")
    parser.add_argument("--report-json", help="Write a JSON report to the given path.")
    parser.add_argument("--report-md", help="Write a Markdown report to the given path.")
    parser.add_argument("--whitelist", help="Optional JSON file listing allowed drifts by category.")
    parser.add_argument(
        "--python-drift-severity",
        choices=("error", "warn", "ignore"),
        help="Override how Python version drift is treated. Defaults to 'error' in CI and 'warn' locally.",
    )
    args = parser.parse_args()

    if args.write_snapshot:
        snapshot_path = pathlib.Path(args.write_snapshot)
        snapshot_path.write_text(json.dumps(build_snapshot(), indent=2), encoding="utf-8")
        print(f"Wrote snapshot to {snapshot_path}")
        return 0

    report = evaluate_parity(args)

    print(f"Dev/CI parity check — Python {sys.version}")
    print(f"Expected Python base image: {report['python']['expected']}")
    print(f"Python drift severity: {report['python_drift_severity']}")
    print(f"Dependency files tracked: {len(report['dependency_files'])}")
    print(f"Extensions tracked: {len(report['extensions'])}")

    for failure in report["failures"]:
        print(f"[FAIL]  {failure}")
    for warning in report["warnings"]:
        print(f"[WARN]  {warning}")

    if args.report_json:
        pathlib.Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report_md:
        pathlib.Path(args.report_md).write_text(build_markdown_report(report), encoding="utf-8")

    if report["failures"]:
        print(f"Parity check FAILED — {len(report['failures'])} failure(s), {len(report['warnings'])} warning(s).")
        return 1

    if report["warnings"]:
        print(f"Parity check PASSED with {len(report['warnings'])} warning(s).")
    else:
        print("Parity check PASSED — environment is in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
