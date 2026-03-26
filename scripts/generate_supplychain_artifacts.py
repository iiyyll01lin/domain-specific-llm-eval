#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def _load_json(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	return json.loads(path.read_text(encoding="utf-8"))


def _extract_components(sbom: Dict[str, Any]) -> Set[str]:
	components = set()
	for component in sbom.get("components", []):
		name = component.get("name", "unknown")
		version = component.get("version", "unknown")
		components.add(f"{name}@{version}")
	return components


def build_sbom_diff(current_sbom: Dict[str, Any], baseline_sbom: Dict[str, Any]) -> Dict[str, Any]:
	current = _extract_components(current_sbom)
	baseline = _extract_components(baseline_sbom)
	return {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"current_component_count": len(current),
		"baseline_component_count": len(baseline),
		"added_components": sorted(current - baseline),
		"removed_components": sorted(baseline - current),
	}


def write_provenance(path: Path, *, sbom_path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"_type": "https://in-toto.io/Statement/v1",
		"predicateType": "https://slsa.dev/provenance/v1",
		"subject": [{"name": sbom_path.name}],
		"predicate": {
			"buildDefinition": {
				"externalParameters": {
					"repository": os.environ.get("GITHUB_REPOSITORY", "local"),
					"git_sha": os.environ.get("GIT_SHA", "local"),
					"version": os.environ.get("BUILD_VERSION", "dev"),
				}
			},
			"runDetails": {
				"builder": {"id": "github-actions" if os.environ.get("GITHUB_RUN_ID") else "local"},
				"metadata": {
					"run_id": os.environ.get("GITHUB_RUN_ID", "local"),
					"generated_at": datetime.now(timezone.utc).isoformat(),
					"url": f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}".rstrip("/"),
				},
			},
		},
	}
	path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def prune_directory(path: Path, keep: int, *, prefix: str, suffix: str) -> None:
	if not path.exists():
		return
	files = sorted(
		[
			item
			for item in path.iterdir()
			if item.is_file()
			and item.name.startswith(prefix)
			and item.name.endswith(suffix)
		],
		key=lambda item: item.stat().st_mtime,
		reverse=True,
	)
	for file_path in files[keep:]:
		file_path.unlink()


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate SBOM diff and provenance artifacts.")
	parser.add_argument("--sbom", required=True)
	parser.add_argument("--baseline-sbom")
	parser.add_argument("--diff", required=True)
	parser.add_argument("--provenance", required=True)
	parser.add_argument("--retain", type=int, default=5)
	args = parser.parse_args()

	sbom_path = Path(args.sbom)
	baseline_path = Path(args.baseline_sbom) if args.baseline_sbom else None
	diff_path = Path(args.diff)
	provenance_path = Path(args.provenance)

	current_sbom = _load_json(sbom_path)
	baseline_sbom = _load_json(baseline_path) if baseline_path else {}

	diff_path.parent.mkdir(parents=True, exist_ok=True)
	diff_path.write_text(json.dumps(build_sbom_diff(current_sbom, baseline_sbom), indent=2), encoding="utf-8")
	write_provenance(provenance_path, sbom_path=sbom_path)
	prune_directory(diff_path.parent, args.retain, prefix=diff_path.stem.split(".")[0], suffix=diff_path.suffix)
	prune_directory(provenance_path.parent, args.retain, prefix=provenance_path.stem.split(".")[0], suffix=provenance_path.suffix)
	print(f"Wrote diff to {diff_path}")
	print(f"Wrote provenance to {provenance_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())