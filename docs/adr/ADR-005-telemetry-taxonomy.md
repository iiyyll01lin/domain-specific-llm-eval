# ADR-005: Telemetry Taxonomy & Naming Conventions

Status: Accepted  
Date: 2025-09-10  

## 1. Context
Consistent telemetry naming enables reliable dashboards, anomaly detection, and cross-layer correlation (frontend ↔ backend). Current design (§25 taxonomy) outlines preliminary event families but lacks a formal decision on structure, prefixes, and versioning boundaries.

## 2. Problem
Without a governed taxonomy, metric and event sprawl leads to brittle queries, duplicated semantics, and difficulty deprecating legacy signals. Need stable conventions supporting evolution and low-friction addition.

## 3. Decision
Adopt a tiered namespace convention:
`<layer>.<domain>.<entity>.<action>[.<qualifier>]`
Where:
- layer ∈ {ui, svc, ws, eval, kg}
- domain: logical subsystem (kg, ingestion, processing, testset, eval, reporting, ws, auth)
- entity: primary object (graph, run, job, chunk, event)
- action: verb or lifecycle state (created, completed, render, connect, error)
- qualifier (optional): detail (retry, timeout, truncated, degraded)

Metrics (Prometheus) use: `<layer>_<domain>_<entity>_<metric>` with unit suffixes (_seconds, _total, _bytes where applicable).

## 4. Rationale
- Predictability: Query patterns standardize (e.g., ui.kg.graph.render, svc.eval.run.completed).
- Collision Avoidance: Clear segmentation prevents overlapping keys.
- Evolvability: Optional qualifier allows structured extension without breaking base patterns.
- Observability Consistency: Aligns frontend events & backend metrics for trace correlation.

## 5. Alternatives
| Approach                       | Pros                | Cons                      | Verdict                         |
|--------------------------------|---------------------|---------------------------|---------------------------------|
| Ad-hoc growth                  | Zero upfront work   | Unbounded entropy         | Rejected                        |
| Flat global names              | Simple              | Hard grouping & filtering | Rejected                        |
| Hierarchical w/ version prefix | Explicit versioning | Verbose for dashboards    | Deferred (future if churn high) |

## 6. Canonical Examples
Events:
- ui.kg.graph.render
- ui.ws.connection.degraded
- svc.ingestion.job.created
- svc.processing.document.completed
- eval.run.completed
- kg.subgraph.request.truncated

Metrics:
- svc_ingestion_job_duration_seconds (histogram)
- svc_processing_chunks_generated_total (counter)
- eval_run_metric_latency_seconds (summary)
- ui_ws_reconnect_attempts_total (counter via proxy mapping)

## 7. Versioning & Deprecation
- Additive only within a minor version of taxonomy.
- Deprecated events: emit both old & new for 2 sprints; mark old with qualifier `.deprecated`.
- Registry file: `/telemetry/taxonomy.json` authoritative list (validated in CI).

## 8. Governance Process
1. Propose new key via PR updating taxonomy.json + rationale.  
2. CI validates format & uniqueness.  
3. Observability review (SRE) for cardinality risk.  
4. Merge on approval; release notes entry added.  

## 9. Risks & Mitigations
| Risk                           | Impact           | Mitigation                                                |
|--------------------------------|------------------|-----------------------------------------------------------|
| Cardinality explosion (labels) | Metrics cost     | Pre-approval of new labels; deny unrestricted user inputs |
| Drift between code & registry  | Stale dashboards | CI scanner comparing emitted vs registry                  |
| Overly granular actions        | Noise            | Encourage aggregation; threshold review                   |

## 10. Success Criteria
- 100% emitted events present in taxonomy registry after Sprint 6.
- No more than 5% of keys marked deprecated concurrently.
- Dashboard queries rely solely on namespace patterns without manual wildcards for >90% panels.

## 11. Future Extensions
- Taxonomy version field & compatibility matrix.
- Automated changelog generator from registry diffs.
- Multi-tenant scoping prefix if SaaS pivot emerges.

---

