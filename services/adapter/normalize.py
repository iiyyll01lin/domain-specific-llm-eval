"""
Insights Normalization Module (TASK-040).

Translates evaluation artifacts (kpis.json, evaluation_items.json, run metadata)
into the portal-friendly export_summary.json schema understood by the UI.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
EXPORT_SUMMARY_SCHEMA_VERSION = "v1"


def normalize_run_summary(
    *,
    run_id: str,
    testset_id: str,
    kpis: Mapping[str, Any],
    evaluation_item_count: int,
    metrics_version: str,
    html_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    created_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert raw evaluation KPIs into a normalised export_summary dict.

    The returned dict conforms to the portal export_summary_v1 schema used by
    the UI's Reports panel.

    Args:
        run_id: Unique evaluation run identifier.
        testset_id: Testset used in this run.
        kpis: Raw aggregated KPI mapping (output of KPIAggregator).
        evaluation_item_count: Total number of evaluated Q/A items.
        metrics_version: Semver string for the metric plugin set.
        html_path: Optional path to the HTML report.
        pdf_path: Optional path to the PDF report.
        created_at: ISO-8601 string for run start.
        completed_at: ISO-8601 string for run completion.

    Returns:
        export_summary dict ready for JSON serialisation.
    """
    normalised_metrics: List[Dict[str, Any]] = []
    for metric_name, distribution in kpis.get("metrics", {}).items():
        if isinstance(distribution, Mapping):
            normalised_metrics.append(
                {
                    "name": metric_name,
                    "mean": _safe_float(distribution.get("mean")),
                    "p50": _safe_float(distribution.get("p50")),
                    "p95": _safe_float(distribution.get("p95")),
                    "count": int(distribution.get("count", 0)),
                }
            )

    summary: Dict[str, Any] = {
        "schema_version": EXPORT_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "testset_id": testset_id,
        "evaluation_item_count": evaluation_item_count,
        "metrics_version": metrics_version,
        "metrics": normalised_metrics,
        "counts": dict(kpis.get("counts", {})),
    }
    if created_at:
        summary["created_at"] = created_at
    if completed_at:
        summary["completed_at"] = completed_at
    if html_path:
        summary["html_path"] = html_path
    if pdf_path:
        summary["pdf_path"] = pdf_path

    return summary


def write_export_summary(
    summary: Dict[str, Any],
    output_path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    Atomically write *summary* to *output_path* as JSON.

    Uses rename-on-write to avoid partial reads.

    Args:
        summary: Normalised summary dict (from :func:`normalize_run_summary`).
        output_path: Destination file path.
        encoding: File encoding (default utf-8).

    Returns:
        Resolved path of the written file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=dest.parent, encoding=encoding
    ) as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    os.replace(tmp, dest)
    return dest


def load_export_summary(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load and return a previously written export_summary JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return f if f == f else None  # exclude NaN
    except (TypeError, ValueError):
        return None


__all__ = [
    "EXPORT_SUMMARY_SCHEMA_VERSION",
    "normalize_run_summary",
    "write_export_summary",
    "load_export_summary",
]
