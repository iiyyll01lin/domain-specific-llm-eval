"""DriftStore — collects per-run GCR KPI averages from ``kpis.json`` artifacts."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunKPIRecord:
    """KPI averages for a single completed evaluation run."""

    run_id: str
    entity_overlap: Optional[float]
    structural_connectivity: Optional[float]
    hub_noise_penalty: Optional[float]


class DriftStore:
    """Scans an outputs root directory and parses GCR KPI averages per run.

    The outputs directory is expected to contain any sub-tree of timestamped
    directories each containing a ``kpis.json`` file produced by
    :class:`~services.eval.kpi_writer.KPIWriter`.

    Sub-directories are sorted lexicographically (``rglob`` result order),
    which matches the ``YYYYMMDD_HHMMSS`` timestamp naming convention used
    by this project — oldest first.
    """

    def __init__(self, outputs_root: str | Path) -> None:
        self._root = Path(outputs_root)

    def load_records(self) -> List[RunKPIRecord]:
        """Return all parseable run KPI records, sorted oldest-first."""
        records: List[RunKPIRecord] = []
        for kpis_file in sorted(self._root.rglob("kpis.json")):
            try:
                data = json.loads(kpis_file.read_text(encoding="utf-8"))
                run_id = data.get("run_id", kpis_file.parent.name)
                metrics = data.get("metrics", {})
                records.append(
                    RunKPIRecord(
                        run_id=run_id,
                        entity_overlap=_avg(metrics, "entity_overlap"),
                        structural_connectivity=_avg(metrics, "structural_connectivity"),
                        hub_noise_penalty=_avg(metrics, "hub_noise_penalty"),
                    )
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("DriftStore: skipping %s — %s", kpis_file, exc)
        return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg(metrics: dict, key: str) -> Optional[float]:
    entry = metrics.get(key)
    if not isinstance(entry, dict):
        return None
    val = entry.get("average")
    return float(val) if val is not None else None


__all__ = ["DriftStore", "RunKPIRecord"]
