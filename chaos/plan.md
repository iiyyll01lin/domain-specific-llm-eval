# Chaos Drill Plan — TASK-104

**Status**: Draft  
**Date**: 2026-03-06  
**Owner**: platform-reliability@team  
**Target Sprint**: 5  
**ADR References**: ADR-004 (manifest integrity), ADR-001 (microservices)

---

## Objective

Validate that the RAG evaluation pipeline recovers gracefully from transient failures without data corruption, job loss, or silent result degradation.

---

## Scope

| Service             | Port  | Failure Modes Tested         |
|---------------------|-------|------------------------------|
| ingestion-service   | 8001  | restart, slow response       |
| processing-service  | 8002  | restart, dependency down     |
| testset-service     | 8003  | LLM endpoint timeout         |
| eval-service        | 8005  | embedding timeout, restart   |
| kg-service          | 8007  | worker crash, slow disk      |
| ws-service          | 8008  | connection drop, overload    |

---

## Drills

### Drill 1 — Service Restart Mid-Job

**Hypothesis**: A service restart while a job is `running` should not cause data corruption. The job may resurface as `failed` or `pending`, but no artifact should contain partial data.

**Steps**:
1. Submit a processing job (`POST /processing-jobs`).
2. While the job status is `running`, kill the processing service container.
3. Restart the container.
4. Poll `/processing-jobs/{job_id}` until terminal state.

**Pass Criteria**:
- Job reaches `failed` or `completed` (not stuck in `running`).
- If `completed`, output artifact passes manifest checksum (see ADR-004).
- No orphaned temporary files in object store.

**Rollback**: N/A (stateless container restart).

---

### Drill 2 — LLM Endpoint Timeout (Testset Generation)

**Hypothesis**: A testset job with a slow LLM endpoint should respect the configured timeout and mark the job `failed` with a descriptive error — not hang indefinitely.

**Steps**:
1. Configure `LLM_ENDPOINT` to a slow proxy (e.g., `tc netem` adds 60 s latency).
2. Submit a testset job (`POST /testset-jobs`).
3. Wait for timeout (default: `LLM_REQUEST_TIMEOUT_SEC`).

**Pass Criteria**:
- Job status transitions to `failed` within `LLM_REQUEST_TIMEOUT_SEC + 5 s`.
- Error message contains "timeout".
- No partial testset file written.

**Rollback**: Remove traffic shaping: `tc qdisc del dev lo root`.

---

### Drill 3 — Embedding Service Unavailable (Eval)

**Hypothesis**: Eval service should fail individual questions gracefully and aggregate remaining results, not crash the entire job.

**Steps**:
1. Block embedding endpoint via iptables: `iptables -I OUTPUT -d <embedding_host> -j DROP`.
2. Submit an eval job with a testset of ≥5 questions.
3. Observe per-question error handling.

**Pass Criteria**:
- Job status reaches `completed` or `partial_failed` (not `running` forever).
- Partial results file is created and references failed question IDs.
- Manifest checksum is valid for the partial result.
- Restore: `iptables -D OUTPUT -d <embedding_host> -j DROP`.

---

### Drill 4 — WebSocket Gateway Overload

**Hypothesis**: Under message burst, the WS gateway should drop connections gracefully (heartbeat miss) rather than OOM-crashing.

**Steps**:
1. Open 100 simultaneous WebSocket connections to `/ui/events`.
2. Do not send any pong responses (simulate unresponsive clients).
3. Wait for `HEARTBEAT_MISS_THRESHOLD` (2 misses × 15 s = 30 s).

**Pass Criteria**:
- All 100 connections closed with code 1001 or 4000 within 45 s.
- ws-service process memory stays below 256 MB.
- New connections are accepted immediately after.

---

### Drill 5 — Object Store Latency (Processing)

**Hypothesis**: Processing service should not silently swallow chunk write errors caused by a slow object store.

**Steps**:
1. Inject 5 s latency on MinIO via `tc netem delay 5000ms`.
2. Submit a processing job.
3. Monitor response time and final artifact.

**Pass Criteria**:
- If job succeeds, all chunks present and checksums valid.
- If job fails, error is surfaced in job status — not silently dropped.
- `http_request_duration_seconds` histogram reflects the induced latency.

---

## Execution Schedule

| Drill | Frequency   | Environment | Automated? |
|-------|-------------|-------------|------------|
| 1     | Per release | staging     | Semi (manual kill step) |
| 2     | Per release | staging     | Yes (tc netem scriptable) |
| 3     | Per release | staging     | Yes (iptables scriptable) |
| 4     | Quarterly   | staging     | Yes (websocket stress script) |
| 5     | Per release | staging     | Yes (tc netem) |

---

## Recovery Runbook References

- [Ingestion Runbook](docs/runbooks/ingestion.md) *(Sprint 6)*
- [Processing Runbook](docs/runbooks/processing.md) *(Sprint 6)*
- [Eval Runbook](docs/runbooks/eval.md) *(Sprint 6)*

---

## Success Criteria Summary

| Criterion                                    | Pass condition                           |
|----------------------------------------------|------------------------------------------|
| No data corruption after service restart     | Manifest checksum valid or job failed    |
| LLM timeout propagated                       | Job failed ≤ timeout + 5 s              |
| Partial eval results preserved               | Partial result file + valid manifest     |
| WS connections closed on heartbeat miss      | All closed within 45 s                  |
| Latency injection visible in metrics         | Histogram p95 reflects injected delay   |
