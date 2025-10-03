from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

from services.eval.aggregation.distribution import MetricDistribution, calculate_distribution
from services.eval.aggregation.sanitizer import sanitize_metrics
from services.eval.aggregation_metrics import AggregationMetricsRecorder


@dataclass(frozen=True)
class AggregatedMetric:
    name: str
    distribution: MetricDistribution


@dataclass(frozen=True)
class KPIAggregationResult:
    metrics: MutableMapping[str, MetricDistribution]
    counts: MutableMapping[str, int]


class KPIAggregator:
    def __init__(
        self,
        *,
        metrics_recorder: Optional[AggregationMetricsRecorder] = None,
    ) -> None:
        self._metrics_recorder = metrics_recorder or AggregationMetricsRecorder()

    def aggregate(self, records: Iterable[Mapping[str, float | None]]) -> KPIAggregationResult:
        start = time.perf_counter()
        sanitized = [sanitize_metrics(record) for record in records]
        metric_names: set[str] = set()
        for record in sanitized:
            metric_names.update(record.keys())

        aggregated: MutableMapping[str, MetricDistribution] = {}
        total_values = 0
        missing_values = 0
        for name in sorted(metric_names):
            raw_values = [record.get(name) for record in sanitized]
            missing_values += sum(1 for value in raw_values if value is None)
            values = [value for value in raw_values if value is not None]
            if not values:
                continue
            distribution = calculate_distribution(values)
            total_values += distribution.count
            aggregated[name] = distribution

        duration = time.perf_counter() - start
        self._metrics_recorder.record(
            item_count=len(sanitized),
            metric_count=len(aggregated),
            duration_seconds=duration,
        )
        counts: MutableMapping[str, int] = {
            "records": len(sanitized),
            "metrics": len(aggregated),
            "values": total_values,
            "missing_values": missing_values,
        }
        return KPIAggregationResult(metrics=aggregated, counts=counts)


__all__ = ["KPIAggregator", "KPIAggregationResult", "AggregatedMetric"]
