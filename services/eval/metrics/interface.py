"""Metric plugin contract (TASK-032a).

This module defines the canonical interface used by the evaluation service to
load and execute metric plugins. Keeping the contract centralised allows us to
validate third-party plugins before they are executed, reducing the likelihood
of runtime failures inside the evaluation loop.

Key concepts
============

``MetricInput``
    Immutable payload describing a single evaluation sample. It contains the
    generated answer, the reference answer (when available), contextual
    artefacts, and trace metadata required for observability.

``MetricValue``
    Normalised metric output produced by a plugin. A plugin may emit multiple
    values if it calculates more than one score (for example precision and
    recall variants).

``MetricPlugin``
    Protocol each concrete plugin must satisfy. Plugins are expected to expose
    a human-readable ``name`` and semantic ``version`` string alongside an
    ``evaluate`` method returning metric values.

``validate_plugin``
    Helper used by the registry/loader to ensure a plugin definition is
    complete. It raises :class:`MetricPluginDefinitionError` with explicit
    messaging to simplify debugging when a required attribute or method is
    missing.

The overall contract version is exported as ``METRIC_PLUGIN_CONTRACT_VERSION``
so that registry endpoints and external tooling can expose it for
compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Type

from services.eval.rag_interface import RetrievedContext

METRIC_PLUGIN_CONTRACT_VERSION = 1


class MetricPluginDefinitionError(ValueError):
    """Raised when a metric plugin definition violates the contract."""


@dataclass(frozen=True)
class MetricInput:
    """Normalised inputs forwarded to metric plugins."""

    run_id: str
    sample_id: str
    question: str
    answer: str
    reference_answer: Optional[str]
    contexts: Tuple[RetrievedContext, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    raw_response: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_iterable_contexts(
        cls,
        *,
        run_id: str,
        sample_id: str,
        question: str,
        answer: str,
        reference_answer: Optional[str],
        contexts: Iterable[RetrievedContext],
        metadata: Mapping[str, object] | None = None,
        raw_response: Mapping[str, object] | None = None,
    ) -> "MetricInput":
        """Convenience constructor accepting any iterable of contexts."""

        return cls(
            run_id=run_id,
            sample_id=sample_id,
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            contexts=tuple(contexts),
            metadata=metadata or {},
            raw_response=raw_response or {},
        )


@dataclass(frozen=True)
class MetricValue:
    """Represents a single metric output computed by a plugin."""

    key: str
    value: float
    confidence: Optional[float] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class MetricPlugin(Protocol):
    """Protocol implemented by all metric plugins."""

    name: str
    version: str

    def evaluate(self, sample: MetricInput) -> Sequence[MetricValue]:
        """Compute one or more metric values for the provided sample."""


def validate_plugin(plugin_cls: Type[MetricPlugin]) -> Type[MetricPlugin]:
    """Validate a plugin definition and return it for fluent usage.

    Args:
        plugin_cls: The plugin class implementing :class:`MetricPlugin`.

    Returns:
        The original ``plugin_cls`` to enable decorator-style usage.

    Raises:
        MetricPluginDefinitionError: If required attributes or methods are
            missing, or if the plugin cannot be instantiated without
            arguments.
    """

    required_attributes = ("name", "version", "evaluate")
    for attribute in required_attributes:
        if not hasattr(plugin_cls, attribute):
            raise MetricPluginDefinitionError(
                f"{plugin_cls.__name__} is missing required attribute '{attribute}'"
            )

    try:
        instance = plugin_cls()  # type: ignore[call-arg]
    except TypeError as exc:  # pragma: no cover - exercised in tests
        raise MetricPluginDefinitionError(
            f"{plugin_cls.__name__} must be instantiable without arguments"
        ) from exc

    name = getattr(instance, "name", None)
    version = getattr(instance, "version", None)
    evaluate = getattr(instance, "evaluate", None)

    if not isinstance(name, str) or not name:
        raise MetricPluginDefinitionError(
            f"{plugin_cls.__name__}.name must be a non-empty string"
        )
    if not isinstance(version, str) or not version:
        raise MetricPluginDefinitionError(
            f"{plugin_cls.__name__}.version must be a non-empty string"
        )
    if not callable(evaluate):
        raise MetricPluginDefinitionError(
            f"{plugin_cls.__name__} must define an 'evaluate' method"
        )

    result = instance.evaluate(  # type: ignore[arg-type]
        MetricInput(
            run_id="_contract_probe",
            sample_id="_contract_probe",
            question="",
            answer="",
            reference_answer=None,
            contexts=tuple(),
        )
    )
    if not isinstance(result, Sequence):
        raise MetricPluginDefinitionError(
            f"{plugin_cls.__name__}.evaluate must return a sequence of MetricValue"
        )
    for item in result:
        if not isinstance(item, MetricValue):
            raise MetricPluginDefinitionError(
                f"{plugin_cls.__name__}.evaluate returned invalid element of type {type(item)!r}"
            )

    return plugin_cls


__all__ = [
    "METRIC_PLUGIN_CONTRACT_VERSION",
    "MetricInput",
    "MetricValue",
    "MetricPlugin",
    "MetricPluginDefinitionError",
    "validate_plugin",
]
