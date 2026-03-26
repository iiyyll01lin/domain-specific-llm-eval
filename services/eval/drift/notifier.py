"""Slack / Teams incoming-webhook notifier for GCR drift alerts.

The webhook URL is read **exclusively** from the ``SLACK_WEBHOOK_URL``
environment variable. If the variable is unset or empty the function is a
no-op — the caller is never forced to handle a missing-key error.

The Slack ``text`` payload uses mrkdwn formatting for readability in both
the desktop and mobile apps.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.eval.drift.detector import DriftResult

logger = logging.getLogger(__name__)

_METRIC_LABELS = {
    "entity_overlap": "Entity Overlap (Sₑ)",
    "structural_connectivity": "Structural Connectivity (Sᶜ)",
    "hub_noise_penalty": "Hub Noise (Pₕ)",
}


def fire_slack_alert(drift_result: "DriftResult") -> bool:
    """POST a formatted alert to a Slack incoming-webhook URL.

    Returns ``True`` on successful HTTP delivery, ``False`` otherwise
    (including when ``SLACK_WEBHOOK_URL`` is not configured).

    The function is deliberately resilient — it will never raise; failures
    are logged at ERROR level.
    """
    url: str = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
        return False

    try:
        import httpx  # deferred: keeps module importable without httpx installed

        flagged = [s for s in drift_result.metrics.values() if s.flagged]
        lines = [
            f"*🚨 Data Drift Alert — `{drift_result.status}`*",
            "",
        ]
        for s in flagged:
            label = _METRIC_LABELS.get(s.metric, s.metric)
            sign = "+" if s.delta_pct >= 0 else ""
            lines.append(
                f"• *{label}*: recent `{s.recent_mean:.3f}` vs "
                f"baseline `{s.baseline_mean:.3f}` "
                f"({sign}{s.delta_pct:.1f}%, z={s.z_score:.2f})"
            )
        lines += [
            "",
            (
                "⚠️ *Action Required:* Analyze failing queries and inject "
                "new domain documents into the Knowledge Graph."
            ),
            f"_Checked at: {drift_result.checked_at}_",
        ]

        payload = {"text": "\n".join(lines)}
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Drift alert posted to Slack (HTTP %d)", resp.status_code)
        return True

    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post Slack drift alert: %s", exc)
        return False


__all__ = ["fire_slack_alert"]
