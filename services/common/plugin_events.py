from __future__ import annotations

from typing import Iterable, Sequence

from services.common.events import EventPublisher
from services.common.metrics import registry

PLUGIN_EVENT_TOTAL = registry.counter(
	name="plugin_events_total",
	help="Plugin lifecycle events by status and kind.",
	labels=["status", "kind", "plugin_name"],
)


def emit_plugin_loaded(
	plugin_name: str,
	*,
	kind: str,
	source: str,
	capabilities: Sequence[str] | None = None,
	publisher: EventPublisher | None = None,
) -> None:
	PLUGIN_EVENT_TOTAL.inc(
		{"status": "loaded", "kind": kind, "plugin_name": plugin_name}
	)
	(publisher or EventPublisher()).publish(
		"plugin.loaded",
		{
			"plugin_name": plugin_name,
			"kind": kind,
			"source": source,
			"capabilities": list(capabilities or []),
		},
	)


def emit_plugin_failed(
	plugin_name: str,
	*,
	kind: str,
	reason: str,
	message: str,
	publisher: EventPublisher | None = None,
) -> None:
	PLUGIN_EVENT_TOTAL.inc(
		{"status": "failed", "kind": kind, "plugin_name": plugin_name}
	)
	(publisher or EventPublisher()).publish(
		"plugin.failed",
		{
			"plugin_name": plugin_name,
			"kind": kind,
			"reason": reason,
			"message": message,
		},
	)


__all__ = ["PLUGIN_EVENT_TOTAL", "emit_plugin_failed", "emit_plugin_loaded"]