from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_supplychain_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("root_generate_supplychain_artifacts", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_sbom_diff = _MODULE.build_sbom_diff
prune_directory = _MODULE.prune_directory
write_provenance = _MODULE.write_provenance


def test_build_sbom_diff_detects_component_changes() -> None:
	current = {"components": [{"name": "a", "version": "1.0.0"}, {"name": "b", "version": "2.0.0"}]}
	baseline = {"components": [{"name": "a", "version": "1.0.0"}]}

	diff = build_sbom_diff(current, baseline)

	assert diff["added_components"] == ["b@2.0.0"]
	assert diff["removed_components"] == []


def test_write_provenance_writes_statement(tmp_path: Path) -> None:
	provenance_path = tmp_path / "provenance.intoto.jsonl"
	write_provenance(provenance_path, sbom_path=Path("sbom-main.json"))

	payload = json.loads(provenance_path.read_text(encoding="utf-8").strip())

	assert payload["_type"] == "https://in-toto.io/Statement/v1"
	assert payload["subject"][0]["name"] == "sbom-main.json"


def test_prune_directory_keeps_latest_files(tmp_path: Path) -> None:
	for index in range(4):
		path = tmp_path / f"file-{index}.json"
		path.write_text(str(index), encoding="utf-8")
	prune_directory(tmp_path, keep=2, prefix="file-", suffix=".json")

	remaining = sorted(path.name for path in tmp_path.iterdir())
	assert len(remaining) == 2