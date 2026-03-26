"""DriftDetector — GCR metric topological drift detection.

Algorithm (Welch's one-sample Z-test per metric)
-------------------------------------------------
Given a **baseline window** B = {b₁…bN} and a **recent window** R = {r₁…rK}:

1. Compute baseline moments:
       μ_B = mean(B),  σ_B = stdev(B)

2. Compute recent rolling mean:
       x̄_R = mean(R)

3. Z-score (standard errors from baseline):
       z = (x̄_R − μ_B) / (σ_B / √K)

4. Directional flag (threshold θ, default 2.0):
       • Sₑ, Sᶜ (higher-is-better) → flagged when z < −θ
       • Pₕ (lower-is-better)       → flagged when z > +θ

5. Severity roll-up:
       0 flags → HEALTHY
       1 flag  → WARNING
       2+ flags → DRIFTING
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from services.eval.drift.store import RunKPIRecord

logger = logging.getLogger(__name__)

_DEFAULT_Z_THRESHOLD: float = 2.0
_DEFAULT_BASELINE_N: int = 100
_DEFAULT_RECENT_K: int = 50
_DEFAULT_MIN_BASELINE: int = 5


@dataclass(frozen=True)
class MetricDriftSummary:
    """Per-metric drift analysis result."""

    metric: str
    baseline_mean: float
    recent_mean: float
    baseline_std: float
    z_score: float
    delta_pct: float  # percentage change relative to baseline mean
    flagged: bool


@dataclass(frozen=True)
class DriftResult:
    """Aggregated drift detection outcome for one check run."""

    status: str  # "HEALTHY" | "WARNING" | "DRIFTING" | "INSUFFICIENT_DATA"
    checked_at: str  # ISO-8601 UTC timestamp
    baseline_window_size: int
    recent_window_size: int
    metrics: Dict[str, MetricDriftSummary] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON responses."""
        return {
            "status": self.status,
            "checked_at": self.checked_at,
            "baseline_window_size": self.baseline_window_size,
            "recent_window_size": self.recent_window_size,
            "message": self.message,
            "metrics": {
                k: {
                    "metric": v.metric,
                    "baseline_mean": v.baseline_mean,
                    "recent_mean": v.recent_mean,
                    "baseline_std": v.baseline_std,
                    "z_score": v.z_score,
                    "delta_pct": v.delta_pct,
                    "flagged": v.flagged,
                }
                for k, v in self.metrics.items()
            },
        }


class DriftDetector:
    """Detect topological data drift in GCR metric time-series.

    Parameters
    ----------
    baseline_n:
        Maximum number of oldest runs that form the baseline window.
    recent_k:
        Number of most recent runs that form the comparison window.
    z_threshold:
        Standard-error distance from the baseline mean at which a metric
        is considered drifted.
    min_baseline:
        Minimum number of baseline samples required before checking.
        Returns ``INSUFFICIENT_DATA`` when fewer records are available.
    """

    def __init__(
        self,
        *,
        baseline_n: int = _DEFAULT_BASELINE_N,
        recent_k: int = _DEFAULT_RECENT_K,
        z_threshold: float = _DEFAULT_Z_THRESHOLD,
        min_baseline: int = _DEFAULT_MIN_BASELINE,
    ) -> None:
        self.baseline_n = baseline_n
        self.recent_k = recent_k
        self.z_threshold = z_threshold
        self.min_baseline = min_baseline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, records: Sequence[RunKPIRecord]) -> DriftResult:
        """Compute current drift status from a sequence of run records.

        Records must be sorted **oldest-first** (``DriftStore`` guarantees this).
        Returns ``INSUFFICIENT_DATA`` when there aren't enough runs yet — this is
        a safe no-alert state; it never raises.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if len(records) < self.min_baseline + 1:
            return DriftResult(
                status="INSUFFICIENT_DATA",
                checked_at=now,
                baseline_window_size=0,
                recent_window_size=len(records),
                message=(
                    f"Need at least {self.min_baseline + 1} runs to form a "
                    f"baseline (have {len(records)})."
                ),
            )

        # Partition: baseline = old runs, recent = latest k runs.
        # split index ensures the baseline never overlaps the recent window.
        split = max(self.min_baseline, len(records) - self.recent_k)
        baseline_records = records[:split][-self.baseline_n:]
        recent_records = records[split:][-self.recent_k:]

        summaries: Dict[str, MetricDriftSummary] = {}
        for attr, direction in (
            ("entity_overlap", "higher"),
            ("structural_connectivity", "higher"),
            ("hub_noise_penalty", "lower"),
        ):
            b_vals = _extract(baseline_records, attr)
            r_vals = _extract(recent_records, attr)
            if len(b_vals) < 2 or not r_vals:
                continue
            summaries[attr] = _compute_summary(
                attr, b_vals, r_vals, direction, self.z_threshold
            )

        flagged_count = sum(1 for s in summaries.values() if s.flagged)
        if flagged_count == 0:
            status = "HEALTHY"
        elif flagged_count == 1:
            status = "WARNING"
        else:
            status = "DRIFTING"

        message = _build_message(status, summaries)
        logger.info("DriftDetector: status=%s flagged=%d", status, flagged_count)

        return DriftResult(
            status=status,
            checked_at=now,
            baseline_window_size=len(baseline_records),
            recent_window_size=len(recent_records),
            metrics=summaries,
            message=message,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract(records: Sequence[RunKPIRecord], attr: str) -> List[float]:
    """Pull non-null, non-NaN float values for *attr* from a list of records."""
    vals: List[float] = []
    for r in records:
        v: Optional[float] = getattr(r, attr, None)
        if v is not None and not math.isnan(v):
            vals.append(v)
    return vals


def _compute_summary(
    metric: str,
    baseline: List[float],
    recent: List[float],
    direction: str,
    z_threshold: float,
) -> MetricDriftSummary:
    b_mean = sum(baseline) / len(baseline)
    # Sample variance (Bessel-corrected)
    variance = sum((x - b_mean) ** 2 for x in baseline) / (len(baseline) - 1)
    b_std = math.sqrt(variance) if variance > 0 else 1e-9

    r_mean = sum(recent) / len(recent)

    # Welch's one-sample Z-score: how many standard errors is the recent
    # cohort away from the baseline mean?
    z = (r_mean - b_mean) / (b_std / math.sqrt(len(recent)))

    delta_pct = ((r_mean - b_mean) / b_mean * 100.0) if b_mean != 0.0 else 0.0

    # Higher-is-better: flag when recent mean drops (z strongly negative).
    # Lower-is-better  : flag when recent mean rises (z strongly positive).
    if direction == "higher":
        flagged = z < -z_threshold
    else:
        flagged = z > z_threshold

    return MetricDriftSummary(
        metric=metric,
        baseline_mean=round(b_mean, 6),
        recent_mean=round(r_mean, 6),
        baseline_std=round(b_std, 6),
        z_score=round(z, 4),
        delta_pct=round(delta_pct, 2),
        flagged=flagged,
    )


_METRIC_LABELS = {
    "entity_overlap": "Entity Overlap (Sₑ)",
    "structural_connectivity": "Structural Connectivity (Sᶜ)",
    "hub_noise_penalty": "Hub Noise (Pₕ)",
}


def _build_message(status: str, summaries: Dict[str, MetricDriftSummary]) -> str:
    flagged = [s for s in summaries.values() if s.flagged]
    if not flagged:
        return "All GCR metrics are within baseline norms."
    parts = [
        f"{_METRIC_LABELS.get(s.metric, s.metric)}: {s.delta_pct:+.1f}% (z={s.z_score:.2f})"
        for s in flagged
    ]
    return f"Data drift detected — {'; '.join(parts)}."


__all__ = ["DriftDetector", "DriftResult", "MetricDriftSummary"]
