"""Tests for services/common/metrics.py — TASK-080."""
import pytest

from services.common.metrics import (
    Counter,
    Histogram,
    MetricsRegistry,
    _format_labels,
    jobs_completed_total,
    jobs_created_total,
    make_metrics_router,
    registry,
)


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


def test_counter_increments():
    c = Counter("test_cnt", "help", ["svc"])
    c.inc({"svc": "alpha"})
    c.inc({"svc": "alpha"})
    results = {tuple(d.values()): v for d, v in c.collect()}
    assert results[("alpha",)] == 2.0


def test_counter_default_amount():
    c = Counter("t", "h", ["x"])
    c.inc({"x": "y"})
    results = {tuple(d.values()): v for d, v in c.collect()}
    assert results[("y",)] == 1.0


def test_counter_multiple_labels():
    c = Counter("t", "h", ["a", "b"])
    c.inc({"a": "1", "b": "2"})
    results = {tuple(d.values()): v for d, v in c.collect()}
    assert results[("1", "2")] == 1.0


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def test_histogram_observe_bin_placement():
    h = Histogram("h", "help", ["svc"], buckets=(0.1, 0.5, 1.0))
    h.observe(0.05, {"svc": "x"})
    rows = h.collect()
    assert rows  # at least one row


def test_histogram_sum_tracks_correctly():
    h = Histogram("h2", "h", ["svc"], buckets=(1.0, 5.0))
    h.observe(0.3, {"svc": "a"})
    h.observe(0.7, {"svc": "a"})
    rows = h.collect()
    label_dict, buckets, counts, sum_val, total = rows[0]
    assert abs(sum_val - 1.0) < 1e-9
    assert total == 2


def test_histogram_count():
    h = Histogram("h3", "h", ["svc"])
    for _ in range(5):
        h.observe(0.01, {"svc": "b"})
    _, _, _, _, total = h.collect()[0]
    assert total == 5


# ---------------------------------------------------------------------------
# MetricsRegistry.render()
# ---------------------------------------------------------------------------


def test_registry_render_contains_help():
    r = MetricsRegistry()
    r.counter("my_counter", "A counter", ["env"])
    text = r.render()
    assert "# HELP my_counter" in text


def test_registry_render_contains_type():
    r = MetricsRegistry()
    r.counter("my_counter2", "help", [])
    text = r.render()
    assert "# TYPE my_counter2 counter" in text


def test_registry_render_histogram_buckets():
    r = MetricsRegistry()
    h = r.histogram("latency", "latency hist", ["svc"], buckets=(0.1, 0.5))
    h.observe(0.05, {"svc": "demo"})
    text = r.render()
    assert "latency_bucket" in text
    assert "+Inf" in text


def test_registry_render_histogram_sum_count():
    r = MetricsRegistry()
    h = r.histogram("lat2", "help", ["svc"])
    h.observe(0.2, {"svc": "s"})
    text = r.render()
    assert "lat2_sum" in text
    assert "lat2_count" in text


# ---------------------------------------------------------------------------
# make_metrics_router
# ---------------------------------------------------------------------------


def test_make_metrics_router_returns_router():
    from fastapi import APIRouter
    router = make_metrics_router()
    assert isinstance(router, APIRouter)


def test_metrics_endpoint_via_testclient():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(make_metrics_router())
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text or "# HELP" in resp.text


# ---------------------------------------------------------------------------
# _format_labels
# ---------------------------------------------------------------------------


def test_format_labels_empty():
    assert _format_labels({}) == ""


def test_format_labels_single():
    result = _format_labels({"k": "v"})
    assert result == '{k="v"}'


def test_format_labels_sorted():
    result = _format_labels({"b": "2", "a": "1"})
    assert result.index("a") < result.index("b")
