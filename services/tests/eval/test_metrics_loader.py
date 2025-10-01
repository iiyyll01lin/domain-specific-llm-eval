from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pytest

from services.eval.metrics.interface import MetricInput, MetricPlugin, MetricValue
from services.eval.metrics.loader import ENTRY_POINT_GROUP, MetricRegistry, load_metric_registry
from services.eval.rag_interface import RetrievedContext


def _sample_input() -> MetricInput:
    return MetricInput(
        run_id="run-1",
        sample_id="sample-1",
        question="Where is the Eiffel Tower located?",
        answer="The Eiffel Tower is located in Paris, France.",
        reference_answer=None,
        contexts=(RetrievedContext(text="The Eiffel Tower is located in Paris, France."),),
    )


class AdditionalMetric:
    name = "additional"
    version = "0.0.1"

    def evaluate(self, sample: MetricInput) -> Sequence[MetricValue]:
        return (MetricValue(key="additional", value=0.5),)


class BrokenMetric:
    name = "broken"
    version = "0.0.1"

    def evaluate(self, sample: MetricInput) -> Sequence[MetricValue]:  # pragma: no cover - executed in tests
        if sample.answer == "The Eiffel Tower is located in Paris, France.":
            raise RuntimeError("boom")
        return (MetricValue(key="broken", value=0.0),)


def test_load_metric_registry_includes_builtin_metrics(monkeypatch) -> None:
    registry = load_metric_registry(entry_point_group=None, additional_plugins=(AdditionalMetric,))
    names = [plugin.name for plugin in registry.plugins]
    assert names == sorted(set(names))
    assert {"faithfulness", "answer_relevancy", "context_precision", "additional"}.issubset(names)


def test_metric_registry_evaluate_isolates_failures(monkeypatch) -> None:
    registry = load_metric_registry(
        builtin_plugins=False,
        additional_plugins=(AdditionalMetric, BrokenMetric),
        entry_point_group=None,
    )

    results = registry.evaluate(_sample_input())

    assert "additional" in results
    assert "broken" not in results


def test_plugin_paths_support(monkeypatch) -> None:
    registry = load_metric_registry(
        builtin_plugins=False,
        plugin_paths=("services.eval.metrics.baseline.faithfulness.FaithfulnessMetric",),
        entry_point_group=None,
    )

    assert any(plugin.name == "faithfulness" for plugin in registry.plugins)


@dataclass
class DummyEntryPoint:
    group: str
    value: type

    def load(self) -> type:
        return self.value


def test_entry_points_loaded(monkeypatch) -> None:
    def fake_entry_points():
        return {ENTRY_POINT_GROUP: [DummyEntryPoint(ENTRY_POINT_GROUP, AdditionalMetric)]}

    monkeypatch.setattr("services.eval.metrics.loader.entry_points", fake_entry_points)

    registry = load_metric_registry(builtin_plugins=False, entry_point_group=ENTRY_POINT_GROUP)
    assert any(plugin.name == "additional" for plugin in registry.plugins)


def test_duplicate_plugin_names_deduplicate(monkeypatch) -> None:
    registry = load_metric_registry(
        builtin_plugins=True,
        additional_plugins=(AdditionalMetric,),
        plugin_paths=("services.eval.metrics.baseline.faithfulness.FaithfulnessMetric",),
        entry_point_group=None,
    )

    names = [plugin.name for plugin in registry.plugins]
    assert names.count("faithfulness") == 1
