from __future__ import annotations

import json
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

from services.common.plugin_events import emit_plugin_failed, emit_plugin_loaded
from services.common.plugin_sandbox import PluginSandboxError, validate_plugin_source

logger = logging.getLogger("extensions")

DEFAULT_EXTENSIONS_DIR = Path(os.environ.get("EXTENSIONS_DIR", "/extensions"))
PLUGIN_CONTRACT_VERSION = 1
_DISCOVERY_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}


def _dev_reload_enabled() -> bool:
	return os.environ.get("EXTENSIONS_DEV_RELOAD", "false").lower() == "true"


def _cache_key(kind: str, directory: Path) -> tuple[str, str]:
	return kind, str(directory.resolve())


def _load_manifest(module_path: Path) -> Dict[str, Any] | None:
	for suffix in (".manifest.json", ".manifest.yaml", ".manifest.yml"):
		manifest_path = module_path.with_suffix(suffix)
		if not manifest_path.exists():
			continue
		if manifest_path.suffix == ".json":
			return json.loads(manifest_path.read_text(encoding="utf-8"))
		try:
			import yaml  # type: ignore[import-not-found]
		except Exception as exc:  # pragma: no cover - optional dependency
			raise ValueError(f"YAML manifest requires PyYAML: {manifest_path}") from exc
		data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
		if not isinstance(data, dict):
			raise ValueError(f"Manifest must be an object: {manifest_path}")
		return data
	return None


def _validate_manifest(module_path: Path, manifest: Dict[str, Any] | None) -> Dict[str, Any] | None:
	if manifest is None:
		return None
	contract_version = manifest.get("contract_version", PLUGIN_CONTRACT_VERSION)
	if contract_version != PLUGIN_CONTRACT_VERSION:
		raise ValueError(
			f"Plugin {module_path.stem} contract_version={contract_version} is incompatible with expected {PLUGIN_CONTRACT_VERSION}"
		)
	capabilities = manifest.get("capabilities", [])
	if not isinstance(capabilities, list):
		raise ValueError(f"Plugin {module_path.stem} capabilities must be a list")
	return manifest


def _load_module(name: str, path: Path) -> ModuleType | None:
	spec = importlib.util.spec_from_file_location(name, path)
	if spec is None or spec.loader is None:
		logger.warning("Invalid plugin spec for %s", path)
		return None

	module = importlib.util.module_from_spec(spec)
	try:
		sys.modules[name] = module
		spec.loader.exec_module(module)
	except Exception as exc:  # pragma: no cover - logged for diagnostics
		logger.error("Failed to import plugin %s: %s", name, exc, exc_info=True)
		return None
	return module


def discover(kind: str, *, directory: Path | None = None) -> Dict[str, Any]:
	"""Discover plugins of the given *kind* in the extensions directory.

	Plugins are Python modules located under ``extensions/``. A module qualifies when
	it defines ``PLUGIN_KIND`` matching *kind*. The loader prefers a callable
	``register`` function (expected to return metadata). If absent, it falls back to
	a ``PLUGIN_DEFINITION`` attribute. Modules that fail to load or return ``None``
	are skipped with a warning.
	"""

	base_dir = directory or DEFAULT_EXTENSIONS_DIR
	key = _cache_key(kind, base_dir)
	if not _dev_reload_enabled() and key in _DISCOVERY_CACHE:
		return dict(_DISCOVERY_CACHE[key])

	if not base_dir.exists():
		logger.info("Extensions directory %s not found; skipping", base_dir)
		return {}

	plugins: Dict[str, Any] = {}
	for module_path in sorted(base_dir.glob("*.py")):
		module_name = f"extensions.{module_path.stem}"
		try:
			validate_plugin_source(module_path)
		except PluginSandboxError as exc:
			emit_plugin_failed(module_path.stem, kind=kind, reason="sandbox", message=str(exc))
			logger.warning("Plugin %s blocked by sandbox: %s", module_name, exc)
			continue

		try:
			manifest = _validate_manifest(module_path, _load_manifest(module_path))
		except ValueError as exc:
			emit_plugin_failed(module_path.stem, kind=kind, reason="manifest", message=str(exc))
			logger.warning("Plugin %s manifest rejected: %s", module_name, exc)
			continue

		module = _load_module(module_name, module_path)
		if module is None:
			emit_plugin_failed(module_path.stem, kind=kind, reason="import", message="module import failed")
			continue

		plugin_kind = getattr(module, "PLUGIN_KIND", None)
		if plugin_kind != kind:
			logger.debug("Skipping plugin %s (kind=%s)", module_name, plugin_kind)
			continue

		register = getattr(module, "register", None)
		payload: Any
		if callable(register):
			try:
				payload = register()
			except Exception as exc:  # pragma: no cover - logged for diagnostics
				emit_plugin_failed(module_path.stem, kind=kind, reason="register", message=str(exc))
				logger.warning("Plugin %s register() failed: %s", module_name, exc, exc_info=True)
				continue
		else:
			payload = getattr(module, "PLUGIN_DEFINITION", None)

		if payload is None:
			logger.debug("Plugin %s provided no payload; skipping", module_name)
			continue

		plugin_name = getattr(module, "PLUGIN_NAME", module_path.stem)
		if isinstance(payload, dict) and manifest is not None:
			payload.setdefault("manifest", manifest)
			payload.setdefault("capabilities", manifest.get("capabilities", []))
		plugins[plugin_name] = payload
		emit_plugin_loaded(
			plugin_name,
			kind=kind,
			source=str(module_path),
			capabilities=list(payload.get("capabilities", [])) if isinstance(payload, dict) else [],
		)
		logger.info("Loaded plugin %s (kind=%s)", plugin_name, kind)

	if not _dev_reload_enabled():
		_DISCOVERY_CACHE[key] = dict(plugins)

	return plugins


__all__ = ["discover", "PLUGIN_CONTRACT_VERSION"]
