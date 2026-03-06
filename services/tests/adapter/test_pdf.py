"""Tests for services.reporting.pdf (TASK-042)."""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from services.reporting.pdf import generate_report, render_template


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": "run-pdf-001",
        "testset_id": "ts-pdf-001",
        "evaluation_item_count": 5,
        "metrics_version": "1.0.0",
        "metrics": [
            {"name": "faithfulness", "mean": 0.9, "p50": 0.91, "p95": 0.95, "count": 5},
        ],
        "counts": {"records": 5},
        "created_at": "2025-01-01T00:00:00Z",
        "completed_at": "2025-01-01T01:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------

def test_render_template_executive_contains_run_id():
    summary = _make_summary()
    html = render_template("executive", summary)
    assert "run-pdf-001" in html


def test_render_template_technical_contains_kpi_json():
    summary = _make_summary()
    kpi_raw = {"faithfulness": {"mean": 0.9}}
    html = render_template("technical", summary, kpi_raw=kpi_raw)
    assert "faithfulness" in html


def test_render_template_unknown_raises():
    summary = _make_summary()
    with pytest.raises(Exception):
        render_template("unknown_template", summary)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_produces_html():
    summary = _make_summary()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_report(
            summary=summary,
            output_dir=tmpdir,
            template="executive",
            generate_pdf=False,
        )
        assert result["html_path"].endswith(".html")
        assert os.path.exists(result["html_path"])
        assert result["pdf_path"] is None


def test_generate_report_technical_template():
    summary = _make_summary()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_report(
            summary=summary,
            output_dir=tmpdir,
            template="technical",
            generate_pdf=False,
        )
        html_content = open(result["html_path"]).read()
    assert "run-pdf-001" in html_content


def test_generate_report_pdf_skipped_when_playwright_unavailable():
    """When Playwright is not installed generate_report should still produce HTML."""
    summary = _make_summary()
    with tempfile.TemporaryDirectory() as tmpdir:
        # html_to_pdf raises RuntimeError when Playwright unavailable
        with patch("services.reporting.pdf.html_to_pdf", side_effect=RuntimeError("playwright not available")):
            result = generate_report(
                summary=summary,
                output_dir=tmpdir,
                template="executive",
                generate_pdf=True,
            )
        assert os.path.exists(result["html_path"])
        assert result["pdf_path"] is None


def test_generate_report_creates_output_dir():
    summary = _make_summary()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "deep", "nested", "dir")
        result = generate_report(
            summary=summary,
            output_dir=nested_dir,
            template="executive",
            generate_pdf=False,
        )
        assert os.path.exists(result["html_path"])
