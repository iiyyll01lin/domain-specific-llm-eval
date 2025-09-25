from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

logger = logging.getLogger("extensions")

DEFAULT_EXTENSIONS_DIR = Path(os.environ.get("EXTENSIONS_DIR", "/extensions"))


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
	if not base_dir.exists():
		logger.info("Extensions directory %s not found; skipping", base_dir)
		return {}

	plugins: Dict[str, Any] = {}
	for module_path in sorted(base_dir.glob("*.py")):
		module_name = f"extensions.{module_path.stem}"
		module = _load_module(module_name, module_path)
		if module is None:
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
				logger.warning("Plugin %s register() failed: %s", module_name, exc, exc_info=True)
				continue
		else:
			payload = getattr(module, "PLUGIN_DEFINITION", None)

		if payload is None:
			logger.debug("Plugin %s provided no payload; skipping", module_name)
			continue

		plugin_name = getattr(module, "PLUGIN_NAME", module_path.stem)
		plugins[plugin_name] = payload
		logger.info("Loaded plugin %s (kind=%s)", plugin_name, kind)

	return plugins


__all__ = ["discover"]
