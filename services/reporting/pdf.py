"""
PDF Report Generation module (TASK-042).

Renders HTML report templates to PDF using Playwright (headless Chromium).
Falls back gracefully when Playwright is not available (CI / lightweight env).

Usage::

    from services.reporting.templates import render_template
    from services.reporting.pdf import html_to_pdf, generate_report

    html = render_template("executive", summary)
    pdf_path = html_to_pdf(html, output_path="reports/run-123/report.pdf")
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

TemplateName = Literal["executive", "technical"]

_PLAYWRIGHT_AVAILABLE: Optional[bool] = None


def _check_playwright() -> bool:
    global _PLAYWRIGHT_AVAILABLE
    if _PLAYWRIGHT_AVAILABLE is None:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            _PLAYWRIGHT_AVAILABLE = True
        except ImportError:
            _PLAYWRIGHT_AVAILABLE = False
    return _PLAYWRIGHT_AVAILABLE


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def render_template(
    template_name: TemplateName,
    summary: Dict[str, Any],
    *,
    kpi_raw: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Render an HTML template with *summary* context.

    Args:
        template_name: ``"executive"`` or ``"technical"``.
        summary: Normalised summary dict from ``adapter.normalize``.
        kpi_raw: Optional raw KPI dict for technical template debug section.

    Returns:
        Rendered HTML string.

    Raises:
        ImportError: If Jinja2 is not installed.
        FileNotFoundError: If the template file is missing.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise ImportError(
            "Jinja2 is required for template rendering. "
            "Install it with: pip install jinja2"
        ) from exc

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(f"{template_name}.html")

    context: Dict[str, Any] = {
        "run_id": summary.get("run_id", "unknown"),
        "testset_id": summary.get("testset_id", "unknown"),
        "evaluation_item_count": summary.get("evaluation_item_count", 0),
        "metrics_version": summary.get("metrics_version", "0.0.0"),
        "metrics": summary.get("metrics", []),
        "counts": summary.get("counts", {}),
        "html_path": summary.get("html_path"),
        "pdf_path": summary.get("pdf_path"),
        "created_at": summary.get("created_at"),
        "completed_at": summary.get("completed_at"),
        "generated_at": _now_iso(),
        "lang": summary.get("lang", "en"),
    }

    if template_name == "technical":
        context["kpi_json"] = json.dumps(
            kpi_raw or {"metrics": summary.get("metrics", [])},
            indent=2,
            ensure_ascii=False,
        )

    return template.render(**context)


# ---------------------------------------------------------------------------
# HTML to PDF conversion
# ---------------------------------------------------------------------------

def html_to_pdf(
    html_content: str,
    output_path: str | os.PathLike[str],
    *,
    viewport_width: int = 1200,
    viewport_height: int = 900,
) -> Path:
    """
    Convert *html_content* to a PDF file at *output_path*.

    Uses Playwright headless Chromium for deterministic rendering.  When
    Playwright is unavailable, raises ``RuntimeError`` with instructions.

    Args:
        html_content: Full HTML document string.
        output_path: Destination path for the PDF.
        viewport_width: Chromium viewport width in pixels.
        viewport_height: Chromium viewport height in pixels.

    Returns:
        Resolved :class:`pathlib.Path` of the written PDF.

    Raises:
        RuntimeError: If Playwright is not installed.
    """
    if not _check_playwright():
        raise RuntimeError(
            "Playwright is required for PDF generation. "
            "Install with: pip install playwright && playwright install chromium"
        )

    from playwright.sync_api import sync_playwright

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write HTML to a temp file so Playwright can load it via file:// URL
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as fh:
        fh.write(html_content)
        tmp_html = Path(fh.name)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": viewport_width, "height": viewport_height}
            )
            page.goto(f"file://{tmp_html.resolve()}", wait_until="networkidle")
            page.pdf(
                path=str(dest),
                format="A4",
                print_background=True,
                margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            )
            browser.close()
    finally:
        tmp_html.unlink(missing_ok=True)

    return dest


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

def generate_report(
    summary: Dict[str, Any],
    output_dir: str | os.PathLike[str],
    template: TemplateName = "executive",
    *,
    generate_pdf: bool = True,
    kpi_raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """
    Render HTML (and optionally PDF) reports and update *summary* with paths.

    Args:
        summary: Normalised summary dict from ``adapter.normalize``.
        output_dir: Directory where report files will be written.
        template: Which template to use (``"executive"`` or ``"technical"``).
        generate_pdf: Whether to attempt PDF generation via Playwright.
        kpi_raw: Optional raw KPI mapping for technical template.

    Returns:
        Dict with keys ``"html_path"`` and ``"pdf_path"`` (``None`` if PDF
        generation was skipped or failed).
    """
    run_id = summary.get("run_id", "unknown")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    html_content = render_template(template, summary, kpi_raw=kpi_raw)
    html_path = out / f"{run_id}_{template}.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("report HTML written", extra={"path": str(html_path), "run_id": run_id})

    pdf_path_str: Optional[str] = None
    if generate_pdf:
        pdf_path = out / f"{run_id}_{template}.pdf"
        try:
            html_to_pdf(html_content, pdf_path)
            pdf_path_str = str(pdf_path)
            logger.info("report PDF written", extra={"path": pdf_path_str, "run_id": run_id})
        except RuntimeError as exc:
            logger.warning(
                "PDF generation skipped: %s",
                exc,
                extra={"run_id": run_id},
            )

    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "report generation completed",
        extra={"run_id": run_id, "duration_ms": duration_ms, "template": template},
    )

    return {"html_path": str(html_path), "pdf_path": pdf_path_str}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


__all__ = [
    "render_template",
    "html_to_pdf",
    "generate_report",
    "TemplateName",
]
