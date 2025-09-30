from typing import Sequence

import pytest

from services.eval.metrics.interface import (
    METRIC_PLUGIN_CONTRACT_VERSION,
    MetricInput,
    MetricPluginDefinitionError,
    MetricValue,
    validate_plugin,
)
from services.eval.rag_interface import RetrievedContext


class ValidMetricPlugin:
    name = "valid"
    version = "1.0.0"

    def evaluate(self, sample: MetricInput) -> Sequence[MetricValue]:
        return [MetricValue(key="score", value=float(len(sample.contexts)))]


class MissingEvaluatePlugin:
    name = "broken"
    version = "0.0.1"


def test_contract_version_is_exported() -> None:
    assert METRIC_PLUGIN_CONTRACT_VERSION == 1


def test_metric_input_from_iterable_contexts() -> None:
    contexts = [RetrievedContext(text="ctx", metadata={"id": 1})]
    metric_input = MetricInput.from_iterable_contexts(
        run_id="run-1",
        sample_id="sample-1",
        question="What is the capital?",
        answer="Paris",
        reference_answer="Paris",
        contexts=contexts,
        metadata={"language": "fr"},
        raw_response={"latency": 0.12},
    )

    assert metric_input.contexts[0].metadata["id"] == 1
    assert metric_input.metadata["language"] == "fr"
    assert metric_input.raw_response["latency"] == 0.12


def test_validate_plugin_accepts_valid_definition() -> None:
    validated = validate_plugin(ValidMetricPlugin)
    instance = validated()

    result = instance.evaluate(
        MetricInput(
            run_id="run",
            sample_id="sample",
            question="Q?",
            answer="A",
            reference_answer=None,
            contexts=tuple(),
        )
    )
    assert result[0].key == "score"


def test_validate_plugin_raises_when_missing_evaluate() -> None:
    with pytest.raises(MetricPluginDefinitionError) as excinfo:
        validate_plugin(MissingEvaluatePlugin)

    assert "evaluate" in str(excinfo.value)
