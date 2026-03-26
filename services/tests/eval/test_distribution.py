from __future__ import annotations

import math

import pytest

from services.eval.aggregation.distribution import MetricDistribution, calculate_distribution


def test_calculate_distribution_basic() -> None:
    distribution = calculate_distribution([1.0, 2.0, 3.0, 4.0])
    assert isinstance(distribution, MetricDistribution)
    assert distribution.count == 4
    assert distribution.minimum == 1.0
    assert distribution.maximum == 4.0
    assert math.isclose(distribution.average, 2.5)
    assert math.isclose(distribution.p50, 2.5)
    assert math.isclose(distribution.p95, 3.85, rel_tol=1e-3)


def test_calculate_distribution_single_value() -> None:
    distribution = calculate_distribution([42.0])
    assert distribution.minimum == 42.0
    assert distribution.maximum == 42.0
    assert distribution.p50 == 42.0
    assert distribution.p95 == 42.0


def test_calculate_distribution_rejects_empty() -> None:
    with pytest.raises(ValueError):
        calculate_distribution([])
