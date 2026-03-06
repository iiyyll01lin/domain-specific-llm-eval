# Eval Service — Operational Runbook

**Service**: `eval` · Port `8005`  
**Owner**: platform-observability@team  
**SLO**: p95 latency < 8 s (per load test threshold in [load/locustfile.py](../../load/locustfile.py)); error rate < 1 %

---

## Overview

The eval service executes the evaluation pipeline: it loads the RAGAS testset, runs metric plugins (faithfulness, answer_relevancy, precision via [metrics/loader.py](../../services/eval/metrics/loader.py)), streams `evaluation_items.json`, and generates the run manifest ([manifest.py](../../services/eval/manifest.py)).

---

## Alert Playbooks

### ALERT: `eval_job_duration_seconds{quantile="0.95"} > 8`

p95 latency has breached SLO.

1. Check if a specific metric plugin is slow:
   ```bash
   kubectl logs -l app.kubernetes.io/name=eval --since=10m | grep "eval_metric_execution_duration"
   ```
2. Look for `METRIC_PLUGIN_FAILED` entries — a failing plugin degrades performance if retried:
   ```bash
   kubectl logs -l app.kubernetes.io/name=eval --since=10m | grep "METRIC_PLUGIN_FAILED"
   ```
3. If LLM judge is backlogged, check the LLM endpoint:
   ```bash
   curl http://adapter/health | jq .llm
   ```
4. Verify rate-limit is not throttling adapters (`RATE_LIMIT_RPM` default 60).

**Escalation**: If unresolved after 20 min, page `platform-observability@team`.

---

### ALERT: `eval_metric_failure_total > 0` (new metric failures)

Individual metric plugins crashed. The remaining plugins still execute (isolation per TASK-032).

1. Identify which plugin failed:
   ```bash
   kubectl logs -l app.kubernetes.io/name=eval | grep "METRIC_PLUGIN_FAILED" | tail -20
   ```
2. Check plugin version in registry:
   ```
   GET http://eval/metrics
   ```
3. If plugin is broken, disable it in config and redeploy.

---

### ALERT: `eval_manifest_mismatch_total > 0`

The evaluation manifest byte check failed — artifact data may be incomplete.

1. Inspect the most recent run manifest:
   ```bash
   cat outputs/run_*/manifests/manifest.json | jq .missing_count
   ```
2. Re-run the failed eval job to regenerate artifacts.

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `eval_job_total{status="success\|failed"}` | Job outcome counters |
| `eval_job_duration_seconds` | End-to-end latency histogram |
| `eval_metrics_registry_load_seconds` | Plugin registry init time |
| `eval_metric_execution_duration_seconds{metric="..."}` | Per-metric execution time |
| `eval_metric_failure_total{metric="..."}` | Per-metric failure counter |
| `eval_evaluation_item_count_total` | Items evaluated in streaming |

Dashboard queries (Prometheus):
```promql
# Plugin failure rate
sum(rate(eval_metric_failure_total[5m])) by (metric)

# p95 eval latency
histogram_quantile(0.95, rate(eval_job_duration_seconds_bucket[5m]))
```

---

## Key Log Codes

| Code | Level | Meaning |
|------|-------|---------|
| `EVAL_JOB_STARTED` | INFO | Eval job picked up |
| `EVAL_JOB_COMPLETE` | INFO | All metrics written |
| `METRIC_PLUGIN_REGISTERED` | INFO | Plugin loaded by registry |
| `METRIC_PLUGIN_FAILED` | ERROR | Plugin execution error (isolated) |
| `EVAL_MANIFEST_MISMATCH` | ERROR | Artifact byte-checksum mismatch |
| `EVAL_JOB_FAILED` | ERROR | Eval job terminated with error |

---

## Runbook: Check Plugin Registry

```bash
# List loaded metric plugins and versions
curl http://eval/metrics | jq '.plugins[] | {name, version}'
```

Expected output for baseline plugins:
```json
{"name": "faithfulness", "version": "1"}
{"name": "answer_relevancy", "version": "1"}
{"name": "precision", "version": "1"}
```

---

## Runbook: Re-run Failed Eval Job

```bash
curl -X POST http://eval/eval-jobs \
  -H 'Content-Type: application/json' \
  -d '{"testset_id": "<testset_id>", "run_id": "<new_run_id>"}'
```

---

## Notes

- Manifest integrity checks run automatically at job completion via `EvaluationManifestBuilder`.
- Run-level artifact manifests (TASK-083) are generated via `generate_run_manifest()` — output at `outputs/run_<id>/manifest.json`.
- Rate limiting (`RATE_LIMIT_RPM`) applies to all `/eval-jobs` endpoints.
