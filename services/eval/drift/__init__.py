"""GCR Data Drift Detection sub-package.

Provides:
- ``DriftStore``    — scans ``outputs/run_*/kpis.json`` for per-run metric averages.
- ``DriftDetector`` — applies Welch's one-sample Z-score to flag drift.
- ``fire_slack_alert`` — posts a formatted alert to a Slack incoming-webhook URL.
- ``create_scheduler`` — returns an APScheduler ``BackgroundScheduler`` wired to
  run drift checks on a configurable interval.
"""

from services.eval.drift.detector import DriftDetector, DriftResult, MetricDriftSummary
from services.eval.drift.notifier import fire_slack_alert
from services.eval.drift.scheduler import create_scheduler, get_last_result
from services.eval.drift.store import DriftStore, RunKPIRecord

__all__ = [
    "DriftDetector",
    "DriftResult",
    "MetricDriftSummary",
    "DriftStore",
    "RunKPIRecord",
    "create_scheduler",
    "get_last_result",
    "fire_slack_alert",
]
