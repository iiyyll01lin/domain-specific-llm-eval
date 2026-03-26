# Processing Service — Operational Runbook

**Service**: `processing` · Port `8002`  
**Owner**: platform-data@team  
**SLO**: p95 latency < 5 s (per load test threshold in [load/locustfile.py](../../load/locustfile.py)); error rate < 1 %

---

## Overview

The processing service consumes ingested CSV rows, applies chunking/embedding strategies, and writes processed chunks to persistent storage. It is the primary CPU-intensive service and supports HPA scaling.

---

## Alert Playbooks

### ALERT: `processing_job_duration_seconds{quantile="0.95"} > 5`

p95 latency has breached SLO.

1. Check whether embedding model is slow:
   ```bash
   kubectl logs -l app.kubernetes.io/name=processing --since=5m | grep "embedding_duration"
   ```
2. If embedding timeouts visible (code `PROC_EMBED_TIMEOUT`):
   - Verify embedding service / sidecar health.
   - See [Chaos Drill 3 — Embedding Unavailable](../../chaos/plan.md#drill-3).
3. If CPU is the bottleneck, scale out:
   ```bash
   kubectl scale deployment <release>-processing --replicas=4
   ```
4. Enable HPA if not already active:
   ```bash
   kubectl patch hpa <release>-processing -p '{"spec":{"minReplicas":2}}'
   ```

**Escalation**: If p95 does not recover within 20 min, page `platform-data@team`.

---

### ALERT: `processing_job_failure_rate > 3%`

1. Look for chunk-persistence errors:
   ```bash
   kubectl logs -l app.kubernetes.io/name=processing --since=10m | grep "PROC_PERSIST_ERR"
   ```
2. Verify storage path is writable (check PVC status if applicable).
3. Check for LLM timeout errors (`PROC_LLM_TIMEOUT`) → see [Chaos Drill 2](../../chaos/plan.md#drill-2).

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `processing_job_total{status="success\|failed"}` | Job outcome counters |
| `processing_job_duration_seconds` | End-to-end job latency histogram |
| `processing_chunk_count_total` | Total chunks produced |
| `processing_embed_duration_seconds` | Embedding call latency |
| `processing_llm_call_total{status="ok\|timeout\|error"}` | LLM call outcomes |

Dashboard queries (Prometheus):
```promql
# p95 latency
histogram_quantile(0.95, rate(processing_job_duration_seconds_bucket[5m]))

# Failure rate
rate(processing_job_total{status="failed"}[5m]) / rate(processing_job_total[5m])
```

---

## Key Log Codes

| Code | Level | Meaning |
|------|-------|---------|
| `PROC_JOB_STARTED` | INFO | Job dequeued and started |
| `PROC_JOB_COMPLETE` | INFO | Chunks written successfully |
| `PROC_EMBED_TIMEOUT` | WARNING | Embedding call exceeded timeout |
| `PROC_LLM_TIMEOUT` | WARNING | LLM call exceeded timeout |
| `PROC_PERSIST_ERR` | ERROR | Chunk write to storage failed |
| `PROC_JOB_FAILED` | ERROR | Job terminated with error |

---

## Runbook: Manual Job Re-queue

For stuck or failed jobs:

```bash
# List failed jobs
curl http://processing/processing-jobs?status=failed | jq '.[].job_id'

# Re-trigger specific job
curl -X POST http://processing/processing-jobs \
  -H 'Content-Type: application/json' \
  -d '{"ingestion_job_id": "<job_id>", "force_reprocess": true}'
```

---

## Scaling Notes

- HPA is configured but **disabled by default** (`autoscaling.enabled: false` in [values.yaml](../../deploy/helm/values.yaml)).
- Enable via `--set processing.autoscaling.enabled=true,processing.autoscaling.maxReplicas=4`.
- Workers are stateless; safe to scale in/out at any time.
