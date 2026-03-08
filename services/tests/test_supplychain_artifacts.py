from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_supplychain_artifacts import build_sbom_diff, prune_directory, write_provenance


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