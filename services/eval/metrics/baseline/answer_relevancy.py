from __future__ import annotations

from services.eval.metrics.baseline._shared import _overlap_ratio, _round
from services.eval.metrics.interface import MetricInput, MetricValue, validate_plugin


@validate_plugin
class AnswerRelevancyMetric:
    name = "answer_relevancy"
    version = "0.1.0"

    def evaluate(self, sample: MetricInput) -> tuple[MetricValue, ...]:
        score, matched, total = _overlap_ratio((sample.answer,), (sample.question,))
        return (
            MetricValue(
                key="answer_relevancy",
                value=_round(score),
                metadata={
                    "matched_answer_tokens": matched,
                    "question_token_total": total,
                },
            ),
        )
