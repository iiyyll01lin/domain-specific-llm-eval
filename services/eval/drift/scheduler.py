"""APScheduler integration for periodic GCR drift checks.

Public surface
--------------
``create_scheduler(outputs_root)``
    Build and return a configured :class:`~apscheduler.schedulers.background.BackgroundScheduler`.
    The caller is responsible for calling ``.start()`` and ``.shutdown()``.

``get_last_result()``
    Return the most-recent :class:`~services.eval.drift.detector.DriftResult`
    (may be ``None`` before the first scheduled check completes).

``run_check_now(outputs_root)``
    Force an immediate synchronous drift check.  Called once at startup so the
    ``/api/v1/drift-status`` endpoint can answer before the first scheduled job.

Environment variables
----------------------
``DRIFT_CHECK_INTERVAL_HOURS``  (default: 6)
    How often the scheduler fires the drift check.  Values below 1 are clamped
    to 1 to prevent runaway polling.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler
    from services.eval.drift.detector import DriftResult

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_HOURS = 6

# Module-level singleton: updated by every scheduled or forced check.
# Intentionally not thread-locked for reads — stale-but-safe reads are fine
# for a best-effort status endpoint.
_last_result: Optional["DriftResult"] = None


def get_last_result() -> Optional["DriftResult"]:
    """Return the most recent drift check result (``None`` until first run)."""
    return _last_result


def run_check_now(outputs_root: str) -> None:
    """Execute a synchronous drift check and update the module-level result."""
    global _last_result
    try:
        from services.eval.drift.store import DriftStore
        from services.eval.drift.detector import DriftDetector
        from services.eval.drift.notifier import fire_slack_alert

        records = DriftStore(outputs_root).load_records()
        result = DriftDetector().evaluate(records)
        _last_result = result
        logger.info("Drift check completed — status=%s runs=%d", result.status, len(records))
        if result.status in ("WARNING", "DRIFTING"):
            fire_slack_alert(result)
    except Exception as exc:  # pragma: no cover
        logger.error("Drift check failed: %s", exc)


def create_scheduler(outputs_root: str) -> "BackgroundScheduler":
    """Create and return a configured APScheduler :class:`BackgroundScheduler`.

    The scheduler is **not started** here; call ``.start()`` inside the
    FastAPI lifespan to ensure it is tied to the application lifecycle.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    try:
        interval_hours = int(
            os.environ.get("DRIFT_CHECK_INTERVAL_HOURS", _DEFAULT_INTERVAL_HOURS)
        )
    except (ValueError, TypeError):
        interval_hours = _DEFAULT_INTERVAL_HOURS

    interval_hours = max(1, interval_hours)

    scheduler: BackgroundScheduler = BackgroundScheduler()
    scheduler.add_job(
        run_check_now,
        trigger="interval",
        hours=interval_hours,
        args=[outputs_root],
        id="drift_check",
        name="GCR Data Drift Check",
        replace_existing=True,
    )
    logger.info(
        "Drift scheduler configured — interval=%dh outputs_root=%s",
        interval_hours,
        outputs_root,
    )
    return scheduler


__all__ = ["create_scheduler", "get_last_result", "run_check_now"]
