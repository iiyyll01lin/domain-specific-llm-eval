# Ingestion Service — Operational Runbook

**Service**: `ingestion` · Port `8001`  
**Owner**: platform-data@team  
**SLO**: p95 latency < 3 s; error rate < 1 %

---

## Overview

The ingestion service receives CSV documents, validates schema, and persists raw rows to the object store before passing job IDs to the processing queue.

---

## Alert Playbooks

### ALERT: `ingestion_job_failure_rate > 5%`

**Triage steps**

1. Check recent logs for schema validation errors:
   ```bash
   kubectl logs -l app.kubernetes.io/name=ingestion --since=10m | grep "INGEST_SCHEMA_ERR"
   ```
2. Verify object-store connectivity:
   ```bash
   curl -s http://ingestion/health | jq .object_store
   ```
3. If `object_store: "degraded"` → see [Object Store Latency drill](../../chaos/plan.md#drill-5).

**Escalation**: If unresolved after 15 min, page `platform-data@team`.

---

### ALERT: `ingestion_queue_depth > 500`

The upstream queue is backed up. Possible causes: slow processing service or high ingest rate.

1. Check processing service health:
   ```
   GET http://processing/health
   ```
2. Scale processing deployment if CPU > 80 %:
   ```bash
   kubectl scale deployment <release>-processing --replicas=3
   ```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `ingestion_job_total{status="success"}` | Total successful ingest jobs |
| `ingestion_job_total{status="failed"}` | Total failed ingest jobs |
| `ingestion_row_count_total` | Total CSV rows ingested |
| `ingestion_job_duration_seconds` | End-to-end job latency histogram |
| `ingestion_schema_error_total` | Schema validation failures |

Dashboard query (Prometheus):
```promql
rate(ingestion_job_total{status="failed"}[5m]) / rate(ingestion_job_total[5m])
```

---

## Key Log Codes

| Code | Level | Meaning |
|------|-------|---------|
| `INGEST_JOB_STARTED` | INFO | New job accepted |
| `INGEST_JOB_COMPLETE` | INFO | Job persisted successfully |
| `INGEST_SCHEMA_ERR` | ERROR | CSV failed schema validation |
| `INGEST_STORE_ERR` | ERROR | Object-store write failed |

---

## Runbook: Forced Re-ingest

If artifacts are corrupted or missing, re-trigger ingest for a job:

```bash
# POST new job with same source file
curl -X POST http://ingestion/ingestion-jobs \
  -H 'Content-Type: application/json' \
  -d '{"source_uri": "<s3://bucket/path/to/file.csv>", "force": true}'
```

---

## Deployment Notes

- Rolling update safe: zero-downtime restart.
- Env var `AUTH_TOKEN` required for authenticated mode.
- The service reads `RATE_LIMIT_RPM` (default 60) from env; set in [ConfigMap](../../deploy/helm/templates/configmap.yaml).
