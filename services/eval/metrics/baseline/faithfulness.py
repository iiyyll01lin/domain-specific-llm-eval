from __future__ import annotations

from services.eval.metrics.baseline._shared import _overlap_ratio, _round
from services.eval.metrics.interface import MetricInput, MetricValue, validate_plugin


@validate_plugin
class FaithfulnessMetric:
    name = "faithfulness"
    version = "0.1.0"

    def evaluate(self, sample: MetricInput) -> tuple[MetricValue, ...]:
        context_texts = tuple(context.text for context in sample.contexts)
        score, matched, total = _overlap_ratio((sample.answer,), context_texts)
        return (
            MetricValue(
                key="faithfulness",
                value=_round(score),
                metadata={
                    "matched_answer_tokens": matched,
                    "answer_token_total": total,
                },
            ),
        )
