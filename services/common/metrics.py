"""TASK-080: Prometheus metrics instrumentation for all services.

Provides a shared ``MetricsRegistry`` and a ``/metrics`` endpoint factory
that can be mounted on any FastAPI application.

Usage:
    from services.common.metrics import make_metrics_router, registry

    app.include_router(make_metrics_router())

Counters & histograms exposed:
- ``http_requests_total{service, method, path, status}``    — per-request counter
- ``http_request_duration_seconds{service, method, path}``  — latency histogram
- ``jobs_created_total{service, job_type}``                 — job creation counter
- ``jobs_completed_total{service, job_type, outcome}``      — job outcome counter (success|failure)
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Minimal in-process metrics (no external library dependency)
# ---------------------------------------------------------------------------


class Counter:
    """A simple monotonic counter."""

    def __init__(self, name: str, help: str, labels: List[str]) -> None:
        self.name = name
        self.help = help
        self.labels = labels
        self._data: Dict[tuple, float] = defaultdict(float)

    def inc(self, labels: Optional[dict] = None, amount: float = 1.0) -> None:
        key = tuple(str(labels.get(l, "")) for l in self.labels) if labels else ()
        self._data[key] += amount

    def collect(self) -> List[tuple]:
        """Return [(label_dict, value), ...]."""
        return [
            ({lk: lv for lk, lv in zip(self.labels, k)}, v)
            for k, v in self._data.items()
        ]


class Histogram:
    """A simple histogram with configurable buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

    def __init__(self, name: str, help: str, labels: List[str], buckets: tuple = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.help = help
        self.labels = labels
        self.buckets = buckets
        self._counts: Dict[tuple, List[int]] = defaultdict(lambda: [0] * (len(buckets) + 1))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, labels: Optional[dict] = None) -> None:
        key = tuple(str(labels.get(l, "")) for l in self.labels) if labels else ()
        self._sums[key] += value
        self._totals[key] += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self._counts[key][i] += 1
        self._counts[key][-1] += 1  # +Inf bucket

    def collect(self) -> List[tuple]:
        result = []
        for key, counts in self._counts.items():
            label_dict = {lk: lv for lk, lv in zip(self.labels, key)}
            result.append((label_dict, self.buckets, counts, self._sums[key], self._totals[key]))
        return result


class MetricsRegistry:
    """Holds all registered metrics and renders a Prometheus text exposition."""

    def __init__(self) -> None:
        self._counters: List[Counter] = []
        self._histograms: List[Histogram] = []

    def counter(self, name: str, help: str, labels: List[str]) -> Counter:
        c = Counter(name=name, help=help, labels=labels)
        self._counters.append(c)
        return c

    def histogram(self, name: str, help: str, labels: List[str], buckets: tuple = Histogram.DEFAULT_BUCKETS) -> Histogram:
        h = Histogram(name=name, help=help, labels=labels, buckets=buckets)
        self._histograms.append(h)
        return h

    def render(self) -> str:
        """Render all metrics in Prometheus text format."""
        lines: List[str] = []

        for c in self._counters:
            lines.append(f"# HELP {c.name} {c.help}")
            lines.append(f"# TYPE {c.name} counter")
            for label_dict, value in c.collect():
                label_str = _format_labels(label_dict)
                lines.append(f'{c.name}{label_str} {value}')

        for h in self._histograms:
            lines.append(f"# HELP {h.name} {h.help}")
            lines.append(f"# TYPE {h.name} histogram")
            for label_dict, buckets, counts, sum_val, total in h.collect():
                label_str_base = _format_labels(label_dict)
                for bound, count in zip(buckets, counts[:-1]):
                    le_labels = _format_labels({**label_dict, "le": str(bound)})
                    lines.append(f'{h.name}_bucket{le_labels} {count}')
                inf_labels = _format_labels({**label_dict, "le": "+Inf"})
                lines.append(f'{h.name}_bucket{inf_labels} {counts[-1]}')
                lines.append(f'{h.name}_sum{label_str_base} {sum_val}')
                lines.append(f'{h.name}_count{label_str_base} {total}')

        return "\n".join(lines) + "\n"


def _format_labels(labels: dict) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


# ---------------------------------------------------------------------------
# Shared global registry & standard metrics
# ---------------------------------------------------------------------------

registry = MetricsRegistry()

http_requests_total = registry.counter(
    name="http_requests_total",
    help="Total HTTP requests",
    labels=["service", "method", "path", "status"],
)

http_request_duration_seconds = registry.histogram(
    name="http_request_duration_seconds",
    help="HTTP request duration in seconds",
    labels=["service", "method", "path"],
)

jobs_created_total = registry.counter(
    name="jobs_created_total",
    help="Total jobs created",
    labels=["service", "job_type"],
)

jobs_completed_total = registry.counter(
    name="jobs_completed_total",
    help="Total jobs completed",
    labels=["service", "job_type", "outcome"],
)


# ---------------------------------------------------------------------------
# FastAPI router factory
# ---------------------------------------------------------------------------


def make_metrics_router():
    """Return a FastAPI APIRouter with a GET /metrics endpoint."""
    from fastapi import APIRouter
    from fastapi.responses import PlainTextResponse

    router = APIRouter()

    @router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics() -> str:  # type: ignore[no-untyped-def]
        return registry.render()

    return router


# ---------------------------------------------------------------------------
# ASGI middleware — records request count & latency automatically
# ---------------------------------------------------------------------------


class MetricsMiddleware:
    """ASGI middleware that records request count and duration."""

    def __init__(self, app, service_name: str) -> None:  # type: ignore[type-arg]
        self.app = app
        self.service_name = service_name

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[type-arg]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = "200"

        async def send_wrapper(message):  # type: ignore[type-arg]
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = str(message.get("status", 200))
            await send(message)

        await self.app(scope, receive, send_wrapper)
        duration = time.perf_counter() - start
        labels = {"service": self.service_name, "method": method, "path": path}
        http_request_duration_seconds.observe(duration, labels=labels)
        http_requests_total.inc(labels={**labels, "status": status_code})
