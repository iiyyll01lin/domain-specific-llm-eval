from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class MetricDistribution:
    count: int
    minimum: float
    maximum: float
    average: float
    p50: float
    p95: float


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty sequence")
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 1:
        return sorted_values[-1]
    index = percentile * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def calculate_distribution(values: Iterable[float]) -> MetricDistribution:
    samples = [float(value) for value in values if not math.isnan(value)]
    if not samples:
        raise ValueError("Metric distribution requires at least one numeric value")
    samples.sort()
    total = sum(samples)
    count = len(samples)
    minimum = samples[0]
    maximum = samples[-1]
    average = total / count
    p50 = _percentile(samples, 0.5)
    p95 = _percentile(samples, 0.95)
    return MetricDistribution(
        count=count,
        minimum=minimum,
        maximum=maximum,
        average=average,
        p50=p50,
        p95=p95,
    )


__all__ = ["MetricDistribution", "calculate_distribution"]
