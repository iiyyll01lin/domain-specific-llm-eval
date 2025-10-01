from __future__ import annotations

from services.eval.metrics.baseline import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
)
from services.eval.metrics.interface import MetricInput
from services.eval.rag_interface import RetrievedContext


def _build_metric_input(
    *,
    question: str,
    answer: str,
    contexts: tuple[RetrievedContext, ...],
) -> MetricInput:
    return MetricInput(
        run_id="run-1",
        sample_id="sample-1",
        question=question,
        answer=answer,
        reference_answer=None,
        contexts=contexts,
    )


def test_faithfulness_metric_reports_full_support() -> None:
    metric = FaithfulnessMetric()
    sample = _build_metric_input(
        question="Where is the Eiffel Tower located?",
        answer="The Eiffel Tower is located in Paris, France.",
        contexts=(
            RetrievedContext(text="The Eiffel Tower is located in Paris, France."),
            RetrievedContext(text="It was constructed in 1889."),
        ),
    )

    (value,) = metric.evaluate(sample)
    assert value.key == "faithfulness"
    assert value.value == 1.0
    assert value.metadata == {"matched_answer_tokens": 8, "answer_token_total": 8}


def test_answer_relevancy_measures_question_overlap() -> None:
    metric = AnswerRelevancyMetric()
    sample = _build_metric_input(
        question="Where is the Eiffel Tower located?",
        answer="The Eiffel Tower is located in Paris, France.",
        contexts=(
            RetrievedContext(text="The Eiffel Tower is located in Paris, France."),
        ),
    )

    (value,) = metric.evaluate(sample)
    assert value.key == "answer_relevancy"
    assert value.value == 0.625
    assert value.metadata == {"matched_answer_tokens": 5, "question_token_total": 8}


def test_context_precision_uses_context_denominator() -> None:
    metric = ContextPrecisionMetric()
    sample = _build_metric_input(
        question="Where is the Eiffel Tower located?",
        answer="The Eiffel Tower is located in Paris, France.",
        contexts=(
            RetrievedContext(text="The Eiffel Tower is located in Paris, France."),
            RetrievedContext(text="It was constructed in 1889."),
        ),
    )

    (value,) = metric.evaluate(sample)
    assert value.key == "context_precision"
    assert value.value == 0.666667
    assert value.metadata == {"matched_context_tokens": 8, "context_token_total": 12}


def test_metrics_are_deterministic() -> None:
    sample = _build_metric_input(
        question="首都在哪裡？",
        answer="台北是台灣的首都。",
        contexts=(
            RetrievedContext(text="台北是台灣的首都，也被稱為臺北。"),
        ),
    )

    metrics = (
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
        ContextPrecisionMetric(),
    )

    results = [metric.evaluate(sample)[0].value for metric in metrics]
    repeat_results = [metric.evaluate(sample)[0].value for metric in metrics]

    assert results == repeat_results


def test_metrics_handle_missing_contexts_gracefully() -> None:
    sample = _build_metric_input(
        question="What is the answer?",
        answer="Unknown.",
        contexts=tuple(),
    )

    metrics = (
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
        ContextPrecisionMetric(),
    )

    for metric in metrics:
        value = metric.evaluate(sample)[0]
        assert 0.0 <= value.value <= 1.0
