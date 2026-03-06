"""Tests for services.adapter.km_export (TASK-045)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from services.adapter.km_export import (
    KG_SUMMARY_SCHEMA,
    TESTSET_SUMMARY_SCHEMA,
    build_kg_summary,
    build_testset_summary,
    compute_summary_delta,
    load_summary,
    write_summary,
)


# ---------------------------------------------------------------------------
# build_testset_summary
# ---------------------------------------------------------------------------

def test_build_testset_summary_fields():
    summary = build_testset_summary(
        testset_id="ts-001",
        sample_count=50,
        seed=42,
        config_hash="abc123",
        persona_count=3,
        scenario_count=5,
    )
    assert summary["schema"] == TESTSET_SUMMARY_SCHEMA
    assert summary["testset_id"] == "ts-001"
    assert summary["sample_count"] == 50
    assert summary["seed"] == 42
    assert summary["config_hash"] == "abc123"
    assert summary["persona_count"] == 3
    assert summary["scenario_count"] == 5
    assert summary["duplicate"] is False
    assert "created_at" in summary


def test_build_testset_summary_duplicate_flag():
    summary = build_testset_summary(
        testset_id="ts-002",
        sample_count=10,
        seed=1,
        config_hash="xyz",
        duplicate=True,
    )
    assert summary["duplicate"] is True


def test_build_testset_summary_custom_created_at():
    ts = "2025-06-01T12:00:00Z"
    summary = build_testset_summary(
        testset_id="ts-003",
        sample_count=20,
        seed=0,
        config_hash="h1",
        created_at=ts,
    )
    assert summary["created_at"] == ts


# ---------------------------------------------------------------------------
# build_kg_summary
# ---------------------------------------------------------------------------

def test_build_kg_summary_fields():
    summary = build_kg_summary(
        kg_id="kg-001",
        node_count=100,
        relationship_count=200,
    )
    assert summary["schema"] == KG_SUMMARY_SCHEMA
    assert summary["kg_id"] == "kg-001"
    assert summary["node_count"] == 100
    assert summary["relationship_count"] == 200
    assert summary["top_entities"] == []
    assert summary["degree_histogram"] == []
    assert summary["histogram_bins"] == 0


def test_build_kg_summary_top_entities_truncated():
    entities = [{"entity": f"e{i}", "frequency": i} for i in range(30)]
    summary = build_kg_summary(
        kg_id="kg-002",
        node_count=50,
        relationship_count=100,
        top_entities=entities,
    )
    assert len(summary["top_entities"]) == 20


def test_build_kg_summary_histogram_truncated():
    histogram = [{"bin": str(i), "count": i} for i in range(60)]
    summary = build_kg_summary(
        kg_id="kg-003",
        node_count=50,
        relationship_count=100,
        degree_histogram=histogram,
    )
    assert len(summary["degree_histogram"]) == 50
    assert summary["histogram_bins"] == 50


# ---------------------------------------------------------------------------
# compute_summary_delta
# ---------------------------------------------------------------------------

def test_compute_summary_delta_no_previous():
    current = {"sample_count": 10}
    delta = compute_summary_delta(current, None)
    assert delta == {}


def test_compute_summary_delta_added():
    current = {"sample_count": 15, "node_count": 100}
    previous = {"sample_count": 10, "node_count": 80}
    delta = compute_summary_delta(current, previous)
    assert delta["delta_sample_count"] == 5
    assert delta["delta_node_count"] == 20


def test_compute_summary_delta_removed():
    current = {"sample_count": 5, "node_count": 60}
    previous = {"sample_count": 10, "node_count": 80}
    delta = compute_summary_delta(current, previous)
    assert delta["delta_sample_count"] == -5
    assert delta["delta_node_count"] == -20


# ---------------------------------------------------------------------------
# write_summary / load_summary
# ---------------------------------------------------------------------------

def test_write_and_load_summary_testset():
    summary = build_testset_summary(
        testset_id="ts-io",
        sample_count=25,
        seed=7,
        config_hash="hash7",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "testset_summary.json")
        write_summary(summary, path)
        assert os.path.exists(path)
        loaded = load_summary(path)
    assert loaded["testset_id"] == "ts-io"
    assert loaded["sample_count"] == 25


def test_write_summary_creates_parent_dirs():
    summary = build_kg_summary(kg_id="kg-io", node_count=5, relationship_count=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "a", "b", "c", "kg_summary.json")
        write_summary(summary, path)
        assert os.path.exists(path)


def test_load_summary_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_summary("/nonexistent/summary/kg.json")
