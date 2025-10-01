from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import EntryPoint, PackageNotFoundError, entry_points
from time import perf_counter
from typing import Iterable, Mapping, MutableMapping, Sequence, Tuple, Type

from prometheus_client import Counter, Histogram

from services.eval.metrics.baseline import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
)
from services.eval.metrics.interface import (
    METRIC_PLUGIN_CONTRACT_VERSION,
    MetricInput,
    MetricPlugin,
    MetricPluginDefinitionError,
    MetricValue,
    validate_plugin,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "rag_eval.metrics"

_REGISTRY_LOAD_SECONDS = Histogram(
    "eval_metrics_registry_load_seconds",
    "Time spent discovering and initialising evaluation metrics",
    labelnames=("source",),
)

_METRIC_DURATION_SECONDS = Histogram(
    "eval_metric_execution_duration_seconds",
    "Time spent executing individual metric plugins",
    labelnames=("metric_name",),
)

_METRIC_FAILURE_TOTAL = Counter(
    "eval_metric_failure_total",
    "Count of metric plugin failures during execution",
    labelnames=("metric_name", "reason"),
)


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    version: str
    plugin: MetricPlugin


def _import_from_path(path: str) -> Type[MetricPlugin]:
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid metric plugin path '{path}'")
    module = import_module(module_name)
    try:
        plugin_cls = getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Metric plugin '{path}' not found") from exc
    return plugin_cls


def _load_entry_points(group: str) -> Sequence[Type[MetricPlugin]]:
    discovered: list[Type[MetricPlugin]] = []
    try:
        eps = entry_points()
    except PackageNotFoundError:  # pragma: no cover - optional dependency
        return discovered

    if isinstance(eps, Mapping):  # Python <3.10 compatibility
        candidates = eps.get(group, [])
    else:
        candidates = eps.select(group=group)

    for ep in candidates:
        if isinstance(ep, EntryPoint):
            plugin_cls = ep.load()
        else:  # pragma: no cover - backwards compatibility
            plugin_cls = ep.load()
        discovered.append(plugin_cls)
    return discovered


def _validate_and_instantiate(
    plugin_cls: Type[MetricPlugin],
) -> MetricPlugin:
    validated = validate_plugin(plugin_cls)
    instance = validated()
    return instance


def _log_plugin_registration(plugin: MetricPlugin, source: str) -> None:
    logger.info(
        "metric.registry.registered",
        extra={
            "code": "METRIC_PLUGIN_REGISTERED",
            "context": {
                "name": getattr(plugin, "name", plugin.__class__.__name__),
                "version": getattr(plugin, "version", "unknown"),
                "source": source,
                "contract_version": METRIC_PLUGIN_CONTRACT_VERSION,
            },
        },
    )


class MetricRegistry:
    def __init__(
        self,
        plugins: Sequence[MetricPlugin],
        *,
        duration_histogram: Histogram = _METRIC_DURATION_SECONDS,
        failure_counter: Counter = _METRIC_FAILURE_TOTAL,
    ) -> None:
        ordered = sorted(plugins, key=lambda plugin: plugin.name)
        self._plugins: Tuple[MetricPlugin, ...] = tuple(ordered)
        self._duration_histogram = duration_histogram
        self._failure_counter = failure_counter

    @property
    def plugins(self) -> Tuple[MetricPlugin, ...]:
        return self._plugins

    def evaluate(self, sample: MetricInput) -> MutableMapping[str, Tuple[MetricValue, ...]]:
        results: MutableMapping[str, Tuple[MetricValue, ...]] = {}
        for plugin in self._plugins:
            start = perf_counter()
            plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
            try:
                raw_values = plugin.evaluate(sample)
                if not isinstance(raw_values, Sequence):
                    raise TypeError(
                        f"Metric plugin '{plugin_name}' returned non-sequence value"
                    )
                tupled = tuple(raw_values)
                for value in tupled:
                    if not isinstance(value, MetricValue):
                        raise TypeError(
                            f"Metric plugin '{plugin_name}' produced invalid element of type {type(value)!r}"
                        )
                results[plugin_name] = tupled
            except Exception as exc:  # pragma: no cover - exercised in tests
                reason = exc.__class__.__name__
                self._failure_counter.labels(metric_name=plugin_name, reason=reason).inc()
                logger.error(
                    "metric.registry.failed",
                    extra={
                        "code": "METRIC_PLUGIN_FAILED",
                        "context": {
                            "metric_name": plugin_name,
                            "reason": reason,
                            "message": str(exc),
                        },
                    },
                )
            finally:
                duration = perf_counter() - start
                self._duration_histogram.labels(metric_name=plugin_name).observe(duration)
        return results


def load_metric_registry(
    *,
    builtin_plugins: bool = True,
    additional_plugins: Iterable[Type[MetricPlugin]] | None = None,
    plugin_paths: Iterable[str] | None = None,
    entry_point_group: str | None = ENTRY_POINT_GROUP,
    registry_load_histogram: Histogram = _REGISTRY_LOAD_SECONDS,
    duration_histogram: Histogram = _METRIC_DURATION_SECONDS,
    failure_counter: Counter = _METRIC_FAILURE_TOTAL,
) -> MetricRegistry:
    start = perf_counter()
    plugin_classes: list[Type[MetricPlugin]] = []

    if builtin_plugins:
        plugin_classes.extend(
            (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextPrecisionMetric,
            )
        )

    if additional_plugins:
        plugin_classes.extend(additional_plugins)

    if plugin_paths:
        for path in plugin_paths:
            plugin_classes.append(_import_from_path(path))

    if entry_point_group:
        plugin_classes.extend(_load_entry_points(entry_point_group))

    loaded_plugins: list[MetricPlugin] = []
    seen_names: set[str] = set()
    for plugin_cls in plugin_classes:
        try:
            plugin = _validate_and_instantiate(plugin_cls)
            name = getattr(plugin, "name", plugin.__class__.__name__)
            if name in seen_names:
                continue
            seen_names.add(name)
            loaded_plugins.append(plugin)
            _log_plugin_registration(plugin, source=plugin_cls.__module__)
        except (MetricPluginDefinitionError, ValueError) as exc:
            logger.error(
                "metric.registry.invalid",
                extra={
                    "code": "METRIC_PLUGIN_FAILED",
                    "context": {
                        "metric_name": getattr(plugin_cls, "__name__", str(plugin_cls)),
                        "reason": exc.__class__.__name__,
                        "message": str(exc),
                    },
                },
            )

    duration = perf_counter() - start
    registry_load_histogram.labels(source="loader").observe(duration)
    return MetricRegistry(
        loaded_plugins,
        duration_histogram=duration_histogram,
        failure_counter=failure_counter,
    )


__all__ = ["MetricRegistry", "load_metric_registry", "ENTRY_POINT_GROUP"]
