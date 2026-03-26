# MLOps Guide — Data Drift Detection, CI/CD, and Monitoring

> **Project:** Domain-Specific RAG Evaluation & MLOps Platform  
> **Version:** 1.1.0  
> **Last updated:** 2026-03-27

---

## Table of Contents

1. [Overview: Why Drift Detection for RAG?](#overview-why-drift-detection-for-rag)
2. [System Architecture — Four Layers](#system-architecture--four-layers)
3. [Layer 1: DriftStore — KPI Persistence Reader](#layer-1-driftstore--kpi-persistence-reader)
4. [Layer 2: DriftDetector — Welch Z-Test Algorithm](#layer-2-driftdetector--welch-z-test-algorithm)
5. [Layer 3: Scheduler — Background APScheduler](#layer-3-scheduler--background-apscheduler)
6. [Layer 4: Notifier — Slack Webhook Integration](#layer-4-notifier--slack-webhook-integration)
7. [Frontend: DriftMonitorBanner Component](#frontend-driftmonitorbannercomponent)
8. [CI/CD: 733-Test Parallel Suite](#cicd-733-test-parallel-suite)
9. [Configuring Drift Detection](#configuring-drift-detection)
10. [Interpreting Alerts](#interpreting-alerts)
11. [Operational Runbook](#operational-runbook)

---

## Overview: Why Drift Detection for RAG?

A RAG system's answer quality depends entirely on the Knowledge Graph (KG) it retrieves from.
When the KG degrades — through stale documents, incomplete ingestion, or domain shift — the
retrieval topology changes *before* users notice hallucinated answers.

The GCR metric family ($S_e$, $S_c$, $P_h$) captures this topology as numeric time series.
**Drift detection applies a statistical test to those time series** to fire an alert the moment
a statistically significant shift occurs — before it becomes a customer-visible incident.

Key properties of the detection system:

- No LLM calls — purely numeric/statistical
- Configurable sensitivity (Z-threshold, window sizes)
- Two-stage severity (`WARNING` → `DRIFTING`) to suppress false-positive pages
- Fully integrated into the Webhook Daemon — zero additional services required

---

## System Architecture — Four Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Webhook Daemon (:8008)                              │
│                                                                          │
│  ┌──────────────┐   ┌──────────────────┐                                │
│  │  DriftStore   │ → │  DriftDetector   │  (Welch Z-test per metric)    │
│  │ (reads KPIs   │   │  (evaluates      │                               │
│  │  from outputs/│   │   time windows)  │                               │
│  │  **/kpis.json)│   └──────────────────┘                               │
│  └──────────────┘          │                                            │
│                            ↓                                            │
│                   ┌──────────────────┐   ┌─────────────────────┐       │
│                   │  APScheduler     │ → │  Slack Notifier      │       │
│                   │  (every 6 hours) │   │  (POST to webhook)   │       │
│                   └──────────────────┘   └─────────────────────┘       │
│                            │                                            │
│                   GET /api/v1/drift-status                              │
│                            ↓                                            │
│                   ┌──────────────────┐                                  │
│                   │ DriftMonitorBanner│  (React component, polls 5min) │
│                   │  in Insights Portal│                                │
│                   └──────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: DriftStore — KPI Persistence Reader

**File:** `services/eval/drift/store.py`

The `DriftStore` does not write any data — it is a **read-only scanner** that discovers
`kpis.json` files in the outputs tree.

### Discovery pattern

```python
for kpis_file in sorted(self._root.rglob("kpis.json")):
    ...
```

Files are scanned via `rglob("kpis.json")` and sorted lexicographically. Because outputs are
named `outputs/run_YYYYMMDD_HHMMSS_*/`, lexicographic order is identical to chronological
order — **oldest runs first**, which the detector requires.

### Expected `kpis.json` format

```json
{
  "run_id": "run_20260327_120000_abc",
  "metrics": {
    "entity_overlap":           { "average": 0.452 },
    "structural_connectivity":  { "average": 0.831 },
    "hub_noise_penalty":        { "average": 0.041 }
  }
}
```

The `RunKPIRecord` dataclass holds the three GCR sub-scores for a single run. Records with
missing or NaN values for individual fields are included with `None` — the detector handles
them gracefully.

---

## Layer 2: DriftDetector — Welch Z-Test Algorithm

**File:** `services/eval/drift/detector.py`

### Algorithm

For each metric $m \in \{S_e, S_c, P_h\}$:

**Step 1:** Partition the sorted run history into two non-overlapping windows:
- **Baseline window** $B$: the oldest N runs (default: up to 100 runs)
- **Recent window** $R$: the most recent K runs (default: last 50 runs)

The partition index guarantees no overlap:
```python
split = max(self.min_baseline, len(records) - self.recent_k)
baseline_records = records[:split][-self.baseline_n:]
recent_records   = records[split:][-self.recent_k:]
```

**Step 2:** Compute baseline moments with Bessel-corrected sample variance:

$$\mu_B = \frac{1}{N}\sum_{i=1}^{N} b_i \ , \quad \sigma_B^2 = \frac{1}{N-1}\sum_{i=1}^{N}(b_i - \mu_B)^2$$

**Step 3:** Compute the Welch one-sample Z-score:

$$z = \frac{\bar{x}_R - \mu_B}{\sigma_B / \sqrt{|R|}}$$

This measures how many **standard errors** the recent cohort mean is from the baseline mean.
Dividing by $\sigma_B / \sqrt{|R|}$ accounts for the sample size of the recent window — larger
windows produce tighter estimates and require smaller absolute shifts to be statistically flagged.

**Step 4:** Direction-aware flagging:

| Metric | Direction | Flag condition |
|--------|-----------|----------------|
| $S_e$ (Entity Overlap) | Higher-is-better | $z < -\theta$ (recent mean fell) |
| $S_c$ (Structural Connectivity) | Higher-is-better | $z < -\theta$ |
| $P_h$ (Hub Noise Penalty) | Lower-is-better | $z > +\theta$ (recent mean rose) |

Default threshold: $\theta = 2.0$ standard errors.

**Step 5:** Severity roll-up:

| Flagged metrics | Status | Meaning |
|-----------------|--------|---------|
| 0 | `HEALTHY` | All metrics within baseline norms |
| 1 | `WARNING` | One metric degraded; monitor closely |
| ≥ 2 | `DRIFTING` | Multiple metrics degraded; immediate investigation required |
| < `min_baseline` records | `INSUFFICIENT_DATA` | Too few runs to form a baseline; safe no-alert state |

### False-positive suppression

Under a null hypothesis (no real drift), each Z-test fires a false positive ~5% of the time
($\theta = 2.0$). By requiring **two independent flags** to trigger `DRIFTING`, the joint
false-positive rate drops to ~0.25%. `WARNING` on a single flag is informational — it does
not trigger a Slack alert by default.

### Guard conditions (zero-division safety)

```python
if sigma_B == 0.0:
    # Regular graph — use epsilon to avoid division by zero
    b_std = 1e-9
```

When all baseline values are identical ($\sigma_B = 0$), the Z-score becomes extremely large
for any recent deviation — correctly flagging even tiny shifts as significant in a perfectly
stable baseline.

---

## Layer 3: Scheduler — Background APScheduler

**File:** `services/eval/drift/scheduler.py`

The scheduler is integrated into the Webhook Daemon's FastAPI lifespan:

```python
# On startup: run immediate check + start background scheduler
run_check_now(outputs_root)  # answers /drift-status before first scheduled job
scheduler = create_scheduler(outputs_root)
scheduler.start()

# On shutdown:
scheduler.shutdown()
```

### Configuration

| Environment variable | Default | Effect |
|---------------------|---------|--------|
| `DRIFT_CHECK_INTERVAL_HOURS` | `6` | Hours between scheduled checks (minimum 1) |

### Immediate check at startup

`run_check_now()` is called once during the lifespan `startup` event, before the first scheduled
job fires. This ensures `GET /api/v1/drift-status` returns a meaningful result immediately rather
than `PENDING` for the first 6 hours.

### Thread safety

The `_last_result` module-level singleton is written by the scheduler thread and read by
FastAPI request handlers. Python's GIL makes the assignment atomic for the purposes of
this best-effort status endpoint — stale-but-safe reads are acceptable here.

---

## Layer 4: Notifier — Slack Webhook Integration

**File:** `services/eval/drift/notifier.py`

### Configuration

```bash
# Set in .env.compose or as a container environment variable
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
```

When `SLACK_WEBHOOK_URL` is unset or empty, `fire_slack_alert()` is a **no-op** — the system
never raises on missing configuration.

### Alert format

Slack messages use `mrkdwn` formatting and include:

```
🚨 Data Drift Alert — `DRIFTING`

• Entity Overlap (Sₑ): recent 0.312 vs baseline 0.452 (-31.0%, z=-4.21)
• Structural Connectivity (Sᶜ): recent 0.601 vs baseline 0.831 (-27.7%, z=-3.88)

⚠️ Action Required: Analyze failing queries and inject new domain documents
into the Knowledge Graph.
_Checked at: 2026-03-27T14:00:00+00:00_
```

### Alert trigger conditions

Alerts fire for `WARNING` and `DRIFTING` status. `HEALTHY` and `INSUFFICIENT_DATA` never
trigger alerts — the system is silent by default and only speaks when action is needed.

### Delivery semantics

- Uses `httpx.post` with a 10-second timeout
- `resp.raise_for_status()` — HTTP errors are caught and logged at ERROR level
- Returns `True` on successful delivery, `False` on any failure
- **Never raises** — alert delivery failure does not affect the drift check result

---

## Frontend: DriftMonitorBanner Component

**File:** `insights-portal/src/components/DriftMonitorBanner.tsx`

The banner polls `GET /api/v1/drift-status` every 5 minutes and renders a status indicator
above the Executive Overview panel.

### Visual states

| Status | Appearance | Default state |
|--------|-----------|---------------|
| `HEALTHY` | Green border, ✅ icon | Collapsed |
| `WARNING` | Amber border, ⚠️ icon, metric detail | **Expanded** |
| `DRIFTING` | Red border, 🚨 icon, metric detail + action CTA | **Expanded** |
| `INSUFFICIENT_DATA` | Grey border, 📊 icon | Collapsed |
| `PENDING` | Grey border, ⏳ icon | Collapsed |
| `UNAVAILABLE` | Grey border, — icon | Collapsed |

### Per-metric detail display

When `WARNING` or `DRIFTING`, the banner renders a breakdown row per flagged metric showing:
- Metric label (e.g., "Structural Connectivity (Sᶜ)")
- Recent mean vs baseline mean
- Delta percentage (colour-coded: red for degradation)
- Z-score
- A `flagged: true/false` badge

### Polling URL resolution

The webhook base URL is resolved from the Vite environment variable:
```typescript
const WEBHOOK_BASE = import.meta.env.VITE_WEBHOOK_BASE_URL ?? 'http://localhost:8008'
```

Override for production deployments by setting `VITE_WEBHOOK_BASE_URL` at build time.

---

## CI/CD: 733-Test Parallel Suite

### Test distribution

| Scope | Tests | Key test files |
|-------|-------|----------------|
| `eval-pipeline/tests/` | 369 | `test_graph_context_relevance.py`, `test_graph_store.py`, `test_drift_*`, `test_v{7-13}_components.py` |
| `services/tests/` | 364 | `test_common_*.py`, `eval/`, `kg/`, `ws/`, `test_validate_dev_parity.py` |
| **Total** | **733** | Collected in 4.10s |

### Parallelism and isolation guarantees

| Mechanism | Guarantee |
|-----------|-----------|
| `pytest-xdist` (`-n auto`) | Each worker process gets a unique temp directory |
| `tmp_path` fixture (function-scoped) | Every test's SQLite DB is in an isolated temp path |
| `asyncio.new_event_loop()` per async test | No shared event loop state between tests |
| `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` | Network calls fail loudly rather than race on shared downloads |

### Running targeted test groups

```bash
# Graph Context Relevance tests only
docker compose -f docker-compose.test.yml run --rm test \
    pytest eval-pipeline/tests/test_graph_context_relevance.py -v

# Drift detection tests only
docker compose -f docker-compose.test.yml run --rm test \
    pytest eval-pipeline/tests/ -k "drift" -v

# Services tests only
docker compose -f docker-compose.test.yml run --rm test \
    pytest services/tests/ -v

# With coverage
docker compose -f docker-compose.test.yml run --rm test \
    pytest -n auto --cov=eval-pipeline/src --cov=services \
    --cov-report=term-missing --cov-report=html
```

---

## Configuring Drift Detection

The `DriftDetector` is configurable at construction time. Production defaults are set via
environment variables on the `webhook` service in `.env.compose`:

| Parameter | Env var | Default | Effect |
|-----------|---------|---------|--------|
| `baseline_n` | — | `100` | Max runs forming the baseline window |
| `recent_k` | — | `50` | Runs in the recent comparison window |
| `z_threshold` | — | `2.0` | Standard errors required to flag a metric |
| `min_baseline` | — | `5` | Minimum runs before checking; returns `INSUFFICIENT_DATA` otherwise |
| Check interval | `DRIFT_CHECK_INTERVAL_HOURS` | `6` | Hours between scheduled checks |
| Slack alerts | `SLACK_WEBHOOK_URL` | (unset) | Slack incoming webhook URL; leave unset to disable |

### Sensitivity tuning guide

| Scenario | Recommendation |
|----------|----------------|
| Very noisy corpus (frequent small fluctuations) | Increase `z_threshold` to 2.5–3.0; increase `recent_k` to 100 |
| High-stakes production (alert early) | Decrease `z_threshold` to 1.8; decrease `recent_k` to 20 |
| Small corpus (< 20 runs) | Decrease `min_baseline` to 3; set `baseline_n` to all available runs |
| On-call fatigue risk | Keep `z_threshold` ≥ 2.5 for `DRIFTING`; accept `WARNING` as info-only |

---

## Interpreting Alerts

### What each flagged metric means

| Metric | Dropped? | Likely cause |
|--------|---------|-------------|
| $S_e$ (Entity Overlap) | ↓ | New documents use different vocabulary; query expansion no longer matches content |
| $S_c$ (Structural Connectivity) | ↓ | KG ingestion failed partially; new chunks not linked to existing nodes |
| $P_h$ (Hub Noise Penalty) | ↑ | New high-connectivity documents were ingested, creating hub nodes that dominate retrieval |

### Recommended response procedure

1. **`WARNING` (single metric):** Review the flagged metric's trend chart in the Insights Portal Analytics view. Check the most recent 10 run outputs for anomalies.

2. **`DRIFTING` (2+ metrics):**
   - Inspect `outputs/run_*/kpis.json` for the flagged run IDs shown in the alert.
   - Run the QA Debugger on the most recent evaluation run to identify failing retrieval patterns.
   - Re-ingest affected document sections via the ingestion service (`POST http://localhost:8001/documents`).
   - After re-ingestion, force an immediate drift check:
     ```bash
     curl -X POST http://localhost:8008/api/v1/drift-check-now
     ```

---

## Operational Runbook

### Check current drift status

```bash
curl -s http://localhost:8008/api/v1/drift-status | python3 -m json.tool
```

### View drift check logs

```bash
docker compose -f docker-compose.services.yml logs webhook --since 1h | grep -i drift
```

### Force immediate drift check

```bash
curl -X POST http://localhost:8008/api/v1/drift-check-now
```

### View all KPI files (inspect drift store inputs)

```bash
find outputs/ -name "kpis.json" | sort | while read f; do
  echo "=== $f ==="; python3 -m json.tool "$f"; echo
done
```

### Reset baseline (discard old run history)

The baseline is derived from the oldest files in `outputs/`. To reset:

```bash
# Archive old runs
mkdir -p outputs/archive
mv outputs/run_2025* outputs/archive/   # adjust glob as needed

# Force a new baseline check
curl -X POST http://localhost:8008/api/v1/drift-check-now
```
