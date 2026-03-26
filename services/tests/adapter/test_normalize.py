"""Tests for services.adapter.normalize (TASK-040)."""
from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from services.adapter.normalize import (
    EXPORT_SUMMARY_SCHEMA_VERSION,
    load_export_summary,
    normalize_run_summary,
    write_export_summary,
)


# ---------------------------------------------------------------------------
# normalize_run_summary
# ---------------------------------------------------------------------------

def _make_kpis(**extra):
    base = {
        "metrics": {
            "faithfulness": {"mean": 0.9, "p50": 0.91, "p95": 0.95, "count": 10},
            "answer_relevancy": {"mean": 0.85, "p50": 0.86, "p95": 0.92, "count": 10},
        },
        "counts": {"records": 10, "missing_values": 0},
    }
    base.update(extra)
    return base


def test_normalize_run_summary_basic():
    kpis = _make_kpis()
    summary = normalize_run_summary(
        run_id="run-001",
        testset_id="ts-001",
        kpis=kpis,
        evaluation_item_count=10,
        metrics_version="1.0.0",
    )

    assert summary["schema_version"] == EXPORT_SUMMARY_SCHEMA_VERSION
    assert summary["run_id"] == "run-001"
    assert summary["testset_id"] == "ts-001"
    assert summary["evaluation_item_count"] == 10
    assert summary["metrics_version"] == "1.0.0"
    assert len(summary["metrics"]) == 2
    names = {m["name"] for m in summary["metrics"]}
    assert names == {"faithfulness", "answer_relevancy"}
    faithfulness = next(m for m in summary["metrics"] if m["name"] == "faithfulness")
    assert math.isclose(faithfulness["mean"], 0.9)
    assert faithfulness["count"] == 10


def test_normalize_run_summary_optional_timestamps():
    kpis = _make_kpis()
    summary = normalize_run_summary(
        run_id="run-002",
        testset_id="ts-002",
        kpis=kpis,
        evaluation_item_count=5,
        metrics_version="1.1.0",
        created_at="2025-01-01T00:00:00Z",
        completed_at="2025-01-01T01:00:00Z",
    )
    assert summary["created_at"] == "2025-01-01T00:00:00Z"
    assert summary["completed_at"] == "2025-01-01T01:00:00Z"


def test_normalize_run_summary_html_pdf_paths():
    kpis = _make_kpis()
    summary = normalize_run_summary(
        run_id="run-003",
        testset_id="ts-003",
        kpis=kpis,
        evaluation_item_count=8,
        metrics_version="1.0.0",
        html_path="/reports/run-003/executive.html",
        pdf_path="/reports/run-003/executive.pdf",
    )
    assert summary["html_path"] == "/reports/run-003/executive.html"
    assert summary["pdf_path"] == "/reports/run-003/executive.pdf"


def test_normalize_run_summary_empty_kpis():
    summary = normalize_run_summary(
        run_id="run-004",
        testset_id="ts-004",
        kpis={},
        evaluation_item_count=0,
        metrics_version="1.0.0",
    )
    assert summary["metrics"] == []
    assert summary["counts"] == {}


def test_normalize_run_summary_non_mapping_metric_skipped():
    kpis = {"metrics": {"faithfulness": "not-a-dict"}, "counts": {}}
    summary = normalize_run_summary(
        run_id="run-005",
        testset_id="ts-005",
        kpis=kpis,
        evaluation_item_count=0,
        metrics_version="1.0.0",
    )
    assert summary["metrics"] == []


# ---------------------------------------------------------------------------
# write_export_summary / load_export_summary
# ---------------------------------------------------------------------------

def test_write_and_load_export_summary():
    kpis = _make_kpis()
    summary = normalize_run_summary(
        run_id="run-w01",
        testset_id="ts-w01",
        kpis=kpis,
        evaluation_item_count=10,
        metrics_version="1.0.0",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "export_summary.json")
        write_export_summary(summary, path)
        assert os.path.exists(path)
        loaded = load_export_summary(path)
    assert loaded["run_id"] == "run-w01"
    assert loaded["schema_version"] == EXPORT_SUMMARY_SCHEMA_VERSION
    assert len(loaded["metrics"]) == 2


def test_write_export_summary_creates_parent_dirs():
    kpis = _make_kpis()
    summary = normalize_run_summary(
        run_id="run-w02",
        testset_id="ts-w02",
        kpis=kpis,
        evaluation_item_count=5,
        metrics_version="1.0.0",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "nested", "summary.json")
        write_export_summary(summary, path)
        assert os.path.exists(path)


def test_load_export_summary_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_export_summary("/nonexistent/path/summary.json")
