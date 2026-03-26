from __future__ import annotations

import json
from pathlib import Path

from services.common.plugin_events import PLUGIN_EVENT_TOTAL
from services.common.plugin_loader import discover


def _write_plugin(
	tmp_path: Path,
	*,
	name: str,
	body: str,
	manifest: dict | None = None,
) -> Path:
	plugin_path = tmp_path / f"{name}.py"
	plugin_path.write_text(body, encoding="utf-8")
	if manifest is not None:
		plugin_path.with_suffix(".manifest.json").write_text(
			json.dumps(manifest),
			encoding="utf-8",
		)
	return plugin_path


def test_discover_loads_manifest_capabilities(tmp_path: Path, monkeypatch) -> None:
	monkeypatch.setenv("EXTENSIONS_DEV_RELOAD", "true")
	_write_plugin(
		tmp_path,
		name="metric_alpha",
		body='PLUGIN_KIND = "metric"\nPLUGIN_NAME = "metric_alpha"\ndef register():\n    return {"name": "metric_alpha"}\n',
		manifest={
			"plugin_name": "metric_alpha",
			"kind": "metric",
			"contract_version": 1,
			"capabilities": ["summary"],
		},
	)

	plugins = discover("metric", directory=tmp_path)

	assert plugins["metric_alpha"]["capabilities"] == ["summary"]


def test_discover_blocks_disallowed_imports(tmp_path: Path, monkeypatch) -> None:
	monkeypatch.setenv("EXTENSIONS_DEV_RELOAD", "true")
	_write_plugin(
		tmp_path,
		name="bad_plugin",
		body='import subprocess\nPLUGIN_KIND = "metric"\nPLUGIN_NAME = "bad_plugin"\ndef register():\n    return {"name": "bad_plugin"}\n',
	)

	plugins = discover("metric", directory=tmp_path)

	assert plugins == {}


def test_discover_skips_incompatible_manifest(tmp_path: Path, monkeypatch) -> None:
	monkeypatch.setenv("EXTENSIONS_DEV_RELOAD", "true")
	_write_plugin(
		tmp_path,
		name="legacy_plugin",
		body='PLUGIN_KIND = "metric"\nPLUGIN_NAME = "legacy_plugin"\ndef register():\n    return {"name": "legacy_plugin"}\n',
		manifest={
			"plugin_name": "legacy_plugin",
			"kind": "metric",
			"contract_version": 99,
			"capabilities": [],
		},
	)

	plugins = discover("metric", directory=tmp_path)

	assert plugins == {}


def test_discover_reflects_add_remove_in_dev_reload(tmp_path: Path, monkeypatch) -> None:
	monkeypatch.setenv("EXTENSIONS_DEV_RELOAD", "true")
	plug = _write_plugin(
		tmp_path,
		name="reloadable_plugin",
		body='PLUGIN_KIND = "metric"\nPLUGIN_NAME = "reloadable_plugin"\ndef register():\n    return {"name": "reloadable_plugin"}\n',
	)

	assert "reloadable_plugin" in discover("metric", directory=tmp_path)
	plug.unlink()
	assert "reloadable_plugin" not in discover("metric", directory=tmp_path)


def test_plugin_events_counter_increments(tmp_path: Path, monkeypatch) -> None:
	monkeypatch.setenv("EXTENSIONS_DEV_RELOAD", "true")
	_write_plugin(
		tmp_path,
		name="counter_plugin",
		body='PLUGIN_KIND = "metric"\nPLUGIN_NAME = "counter_plugin"\ndef register():\n    return {"name": "counter_plugin"}\n',
	)

	before = sum(value for _, value in PLUGIN_EVENT_TOTAL.collect())
	discover("metric", directory=tmp_path)
	after = sum(value for _, value in PLUGIN_EVENT_TOTAL.collect())

	assert after > before