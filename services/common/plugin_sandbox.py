from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable, Set


class PluginSandboxError(RuntimeError):
	"""Raised when a plugin imports modules outside the allowlist."""


DEFAULT_ALLOWED_IMPORTS = {
	"collections",
	"dataclasses",
	"json",
	"math",
	"pathlib",
	"ragas",
	"re",
	"services",
	"statistics",
	"typing",
	"uuid",
}


def allowed_imports() -> Set[str]:
	extra = os.environ.get("PLUGIN_IMPORT_ALLOWLIST", "")
	configured = {item.strip() for item in extra.split(",") if item.strip()}
	return DEFAULT_ALLOWED_IMPORTS | configured


def _iter_import_roots(tree: ast.AST) -> Iterable[str]:
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			for alias in node.names:
				yield alias.name.split(".", 1)[0]
		elif isinstance(node, ast.ImportFrom) and node.module:
			yield node.module.split(".", 1)[0]


def validate_plugin_source(path: Path) -> None:
	tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
	allowed = allowed_imports()
	blocked = sorted({root for root in _iter_import_roots(tree) if root not in allowed})
	if blocked:
		raise PluginSandboxError(
			f"disallowed imports {blocked}; allowed roots: {sorted(allowed)}"
		)


__all__ = ["PluginSandboxError", "validate_plugin_source", "allowed_imports"]