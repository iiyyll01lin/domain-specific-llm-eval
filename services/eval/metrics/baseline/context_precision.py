from __future__ import annotations

from services.eval.metrics.baseline._shared import _context_precision, _round
from services.eval.metrics.interface import MetricInput, MetricValue, validate_plugin


@validate_plugin
class ContextPrecisionMetric:
    name = "context_precision"
    version = "0.1.0"

    def evaluate(self, sample: MetricInput) -> tuple[MetricValue, ...]:
        context_texts = tuple(context.text for context in sample.contexts)
        score, matched, total = _context_precision(context_texts, (sample.answer,))
        return (
            MetricValue(
                key="context_precision",
                value=_round(score),
                metadata={
                    "matched_context_tokens": matched,
                    "context_token_total": total,
                },
            ),
        )
