# UI Technical Design – Evaluation Workflow Console

Version: 0.1  
Status: Draft for review  
Date: 2025-09-10  
Owner: Platform Engineering  

---
## 1. Purpose & Scope
This document translates the UI EARS requirements (`../requirements/requirements.ui.md`) into a concrete technical architecture and implementation strategy for the Evaluation Workflow Console (extension module inside the existing Insights Portal). It defines component boundaries, data & control flows, state management, feature flag integration, error handling patterns, performance strategies, test approach, and future extensibility.

Non‑Goals (Phase 1): Complete RBAC implementation, full offline-first support, derivative CRUD UI, visual polish guidelines, final accessibility test evidence.

## 2. Architectural Overview
### 2.1 High-Level Concept
The console is an embedded module within the existing Portal (React + Vite). It introduces lifecycle navigation (Documents → Processing → KG → Testsets → Evaluations → Insights → Reports → Admin) plus optional Knowledge Graph visualization (flag gated) and multi-run comparison overlays. All backend interaction is via service APIs or normalized artifact fetch (object storage) – no new bespoke backend layer.

### 2.2 Runtime Component Diagram
```
+---------------- Portal Shell (existing) ----------------+
|  Theme/I18n  |  FeatureFlagsProvider  |  Router          |
|              |          |              |                 |
|              v          v              v                 |
|         UI Lifecycle Module (this design)               |
|   +-----------+  +-----------+  +-----------+           |
|   | Documents |  | Processing|  |   KG      | ...       |
|   +-----------+  +-----------+  +-----------+           |
|        |             |              |                   |
|        | REST (fetch)|              | REST + Artifacts  |
|        v             v              v                   |
|  ingestion-svc   processing-svc   kg-builder-svc         |
|        |             |              |                   |
|  object storage  vector store   KG JSON summaries       |
+---------------------------------------------------------+
```

### 2.3 Key Principles
- Thin orchestration: UI does not orchestrate multi-step workflows; it triggers jobs and observes status.
- Declarative state: Central store (Zustand existing portal store) holds run/testset/KG summaries; lists fetch on demand; no over-caching.
- Progressive enhancement: Feature flags lazily mount optional graphs & compare panels.
- Fail fast: API errors surface via unified error drawer and do not block unrelated panels.

## 3. Data Flow & Sequence Diagrams
### 3.1 Submit Processing Job
```
User → UI: click "Start Processing"
UI → ingestion-svc: GET /documents/:id (confirm status)
UI → processing-svc: POST /process-jobs { document_id, profile_hash }
processing-svc → UI: 202 { job_id }
UI → processing-svc: (poll every 10s) GET /process-jobs/:job_id
loop until terminal
processing-svc → UI: { status: completed, chunk_count, embedding_profile_hash }
UI: update table row & emit toast
```

### 3.2 KG Visualization (Flag Path)
```
UI boot → FeatureFlagsProvider: fetch /config/feature-flags
[kgVisualization = true]
User selects KG tab
UI → kg-builder-svc: GET /kg/:id/summary (node_count, relationship_count, degree_histogram, top_entities[])
UI (lazy import): import('cytoscape') & render CytoscapeGraph component
```

### 3.3 Evaluation Run Monitoring
```
User triggers run (outside scope or via future UI form)
UI → eval-runner-svc: GET /eval-runs?filters
UI ↔ eval-runner-svc: poll/WS /eval-runs/:id/stream progress events
On completion: fetch insights adapter artifacts (JSON) and update run detail panel
```

### 3.4 Multi-Run Compare (Flag Path)
```
User toggles compare mode (flag multiRunCompare true)
UI: enable selection checkboxes on run list
UI: compute deltas client-side using normalized KPIs (kpis.json) loaded per run
UI: display delta badges & variance sparkline
```

## 4. Interfaces (External Service Contracts Consumed)
| UI Panel | Service | Endpoint | Method | Payload (Req) | Notable Response Fields |
|----------|---------|----------|--------|---------------|-------------------------|
| Documents | ingestion-svc | /documents | GET (list) | n/a | [{km_id, version, checksum, status}]
| Documents | ingestion-svc | /documents/:id | GET | n/a | {status, size, hash}
| Processing | processing-svc | /process-jobs | POST | {document_id, profile_hash} | 202 {job_id}
| Processing | processing-svc | /process-jobs/:id | GET | n/a | {status, progress, chunk_count}
| KG | kg-builder-svc | /kg | GET | n/a | [{kg_id, node_count, relationship_count, status}]
| KG | kg-builder-svc | /kg/:id/summary | GET | n/a | {node_count, relationship_count, degree_histogram[], top_entities[]}
| Testsets | testset-gen-svc | /testset-jobs | POST | {...config} | 202 {job_id}
| Testsets | testset-gen-svc | /testset-jobs/:id | GET | n/a | {status, sample_count, seed, config_hash}
| Evaluations | eval-runner-svc | /eval-runs | GET | filters | [{run_id, metrics_version, progress, verdict}]
| Evaluations | eval-runner-svc | /eval-runs/:id | GET | n/a | {progress, verdict, error_count}
| Evaluations | eval-runner-svc | /eval-runs/:id/stream | WS | n/a | progress events
| Reports | reporting-svc | /reports | GET | filters | [{report_id, run_id, status, type}]
| Reports | reporting-svc | /reports/:id | GET | n/a | {status, html_url, pdf_url}
| KM Summaries | insights-adapter-svc | /exports | GET | type=km_summary | summary entries
| Feature Flags | config endpoint | /config/feature-flags | GET | n/a | flag booleans & metadata

## 5. Client-Side Data Models (TypeScript)
```ts
export interface DocumentRow { km_id: string; version: string; checksum: string; status: string; size?: number; last_event_ts?: string }
export interface ProcessingJob { job_id: string; document_id: string; status: string; progress: number; chunk_count?: number; embedding_profile_hash?: string }
export interface KnowledgeGraphSummary { kg_id: string; node_count: number; relationship_count: number; degree_histogram?: number[]; top_entities?: { entity: string; degree: number }[] }
export interface TestsetJob { job_id: string; status: string; sample_count?: number; seed?: number; config_hash?: string }
export interface EvaluationRun { run_id: string; testset_id: string; metrics_version: string; progress: number; verdict?: string; error_count: number }
export interface ReportEntry { report_id: string; run_id: string; status: string; type: 'html'|'pdf'|'both'; html_url?: string; pdf_url?: string }
export interface KMSummary { kind: 'testset'|'kg'; schema_version: string; created_at: string; counts: Record<string, number> }
```

## 6. State Management Strategy
- Global store additions: documents[], processingJobs[], kgSummaries[], testsetJobs[], evalRuns[], reports[], kmSummaries[].
- Derived selectors: activeProcessing(document_id), lastKGSummaries(limit), runDelta(baseRunId, compareRunId).
- Normalization: All arrays keyed by id (Map<string, T>) internally for O(1) updates; selectors expose sorted arrays.

## 7. Feature Flag Integration
- Single fetch on boot (Phase 1) via existing `FeatureFlagsProvider` – update provider to call `/config/feature-flags` (server contract) instead of static JSON path.
- Lazy boundaries:
  - KG Visualization component chunk `kg-viz.[hash].js` only imported when flag true and user opens KG tab.
  - Multi-run compare overlay code behind dynamic import for charts if additional libs added later.
- Future (Phase 2): Introduce re-fetch at `refreshIntervalSeconds` if provided.

## 8. Error Handling & Resilience
| Layer    | Pattern                                                                                           | Example                              |
|----------|---------------------------------------------------------------------------------------------------|--------------------------------------|
| Fetch    | Unified wrapper adds trace_id header (future) & parses JSON with size guard                       | fetchJson(url, {maxBytes:2_000_000}) |
| UI       | Error Drawer context collects {ts, service, code, trace_id}                                       | pushError(err) from hooks            |
| Retry    | Non-terminal 5xx on polling endpoints: exponential backoff (1s,2s,4s, max 4) then surface warning | pollWithBackoff                      |
| Fallback | Missing optional fields (degree_histogram) gracefully hide viz not entire panel                   | conditional render                   |
| Timeouts | AbortController with 10s default; aborted requests logged                                         | fetch with signal                    |

## 9. Performance & Bundling
- Initial bundle budget: ≤1.2MB gzip (UI-NFR-006). Track via Vite analyze plugin in CI.
- Code splitting: dynamic imports for KG visualization & multi-run compare modules.
- Rendering strategy: virtualization (react-window) for lists > 500 rows future path; Phase 1 not required.
- Poll throttling: unify polling intervals (documents, processing, evaluations) through a scheduler to ensure ≤1 request/5s/list (UI-FR-050).
- Memoization: derived KPI deltas computed with shallow cache keyed by run_id pair.

## 10. Internationalization (i18n)
- Reuse existing i18n loader; add UI namespace `lifecycle` for new lifecycle strings.
- Missing key handler logs and displays key (UI-FR-048) – implement via i18n fallback interceptor.

## 11. Accessibility Strategy
- Baseline component library already WCAG aligned; new panels follow color tokens.
- Keyboard focus order matches lifecycle tab order; arrow navigation within graph (Phase 2 optional improvement).
- Cytoscape canvas: provide textual summary (node/edge counts + top entities) for screen readers when graph present.

## 12. Testing Strategy
| Layer               | Tests                                                                       |
|---------------------|-----------------------------------------------------------------------------|
| Unit                | hooks (polling, delta calc), feature flag provider fallback, store reducers |
| Component           | KG summary panel (with / without flag), Processing table SLA highlight      |
| Integration         | Multi-run compare selection + delta render, report list fallback to HTML    |
| Contract (Mock)     | Validate expected fields for /kg/:id/summary mapping to model               |
| Performance (Build) | Bundle size assertion & chunk presence/absence tests                        |
| E2E (Playwright)    | Lifecycle tab navigation persistence, flag-enabled KG panel visibility      |

Coverage Targets: critical new modules ≥80% statements.

## 13. Traceability Mapping
Representative links (not exhaustive):
- UI-FR-016/017/018 → KG summary panel + optional viz (flag gate).
- UI-FR-023/024/026 → Evaluation run list + progress polling & error count column.
- UI-FR-029 → Multi-run compare overlay (flag).
- UI-FR-033/034/035 → KM summary table integration.
- UI-FR-049/050/051 → Performance strategies (bundle budget, polling scheduler, event handling latency).
- UI-NFR-006 → Code splitting + lazy import plan.
- UI-NFR-004 → a11y checks integrated via axe in CI (future step).

## 14. Security & Privacy Considerations
- Phase 1: No auth; implement stub token injection function to ease Phase 2 upgrade.
- Masked secrets handling UI-FR-056: central redact util before rendering secret fields.
- PII masking UI-FR-058: `maskPII(record)` applied in artifact ingestion stage pipeline (client-side fallbacks only for visualization, not source persistence).

## 15. Logging & Telemetry
- Structured client log helper: logEvent({type, component, duration_ms, trace_id?}).
- Flag state snapshot logged once at boot (debug) to assist reproduction.
- Error drawer doubles as telemetry feed (could post back in Phase 2).

## 16. Extensibility & Plugin Surface
- Extension loader (dev mode) scans a configurable global (window.__UI_EXTENSIONS__) providing register(panel: PanelDefinition).
- Panels mount inside lifecycle grid with collision avoidance by name.
- Experimental metric visualizers sandboxed in iframe; message channel limited to {type:'metricData', payload:...}.

## 17. Future Enhancements (Roadmap)
| Area    | Enhancement                                       | Rationale                        |
|---------|---------------------------------------------------|----------------------------------|
| Flags   | Signed flag payload & ETag cache                  | Integrity + reduced bandwidth    |
| Graph   | Server-side sampling & clustering pre-aggregation | Performance on large KGs         |
| Offline | Local queue persistence (IndexedDB)               | Resilience for unstable networks |
| Auth    | OAuth2 client credentials integration             | Production security              |
| Replay  | Scenario replay panel with stored prompts         | Debug repeatability              |
| Metrics | Client perf HUD (FID, LCP sampling)               | Frontend performance governance  |

## 18. Open Decisions
| Topic                               | Status                   | Next Step                     |
|-------------------------------------|--------------------------|-------------------------------|
| WebSocket multiplex vs direct       | Pending                  | Benchmark gateway complexity  |
| Offline action scope                | Pending                  | Define idempotent action list |
| Graph summarization shape evolution | Controlled by KG service | Add version field if extended |

## 19. Risks & Mitigations
| Risk                     | Impact         | Mitigation                                       |
|--------------------------|----------------|--------------------------------------------------|
| Flag sprawl unmanaged    | UI complexity  | Central registry + lint rule for unused flags    |
| Excess polling load      | Backend strain | Shared scheduler & exponential backoff on errors |
| Large KG summary payload | Slow render    | Top-N truncation + histogram bins cap (e.g., 50) |
| Bundle creep (viz libs)  | Breach NFR     | Periodic bundle analyze gating CI                |

## 20. Implementation Phases
| Phase | Focus                                                  | Key Deliverables                                                            |
|-------|--------------------------------------------------------|-----------------------------------------------------------------------------|
| 1     | Baseline lifecycle tables + flags provider integration | Documents/Processing/KG/Testsets/Evaluations skeleton + feature flags fetch |
| 2     | Visualization & compare features                       | Cytoscape lazy load, multi-run compare overlay                              |
| 3     | Advanced ops & replay                                  | SLA dashboards, scenario replay, richer KG filtering                        |
| 4     | Hardening & auth                                       | Auth integration, a11y audit, performance tuning                            |

## 21. WebSocket Event Interface (Draft)
### 21.1 Event Channel Strategy
Initial implementation may use polling; this spec defines a future upgrade path to a single multiplex WebSocket endpoint: `wss://<gateway>/ui/events`. The client subscribes with a JSON handshake.

Handshake:
```json
{ "action":"subscribe", "topics":["documents","processing","kg","testsets","eval_runs","reports"], "client_version":"ui-0.1", "trace_id":"<uuid>" }
```

Server → Client Event Envelope:
```json
{
  "topic": "eval_runs",
  "type": "progress.update",
  "ts": "2025-09-10T10:00:12.345Z",
  "trace_id": "...",
  "data": {
    "run_id": "uuid",
    "progress": 42,
    "error_count": 1,
    "partial_metrics": { "faithfulness": 0.81 }
  }
}
```

Topics & Types (MVP):
| Topic | Types | Data Fields |
|-------|-------|-------------|
| documents | status.update | { document_id, status }
| processing | progress.update | { job_id, progress, chunk_count? }
| kg | build.update | { kg_id, status, node_count?, relationship_count? }
| testsets | job.update | { job_id, status, sample_count? }
| eval_runs | progress.update, completed, failed | { run_id, progress, error_count?, verdict? }
| reports | status.update | { report_id, status }

### 21.2 Client Handling
- Single WS connection managed by a hook `useEventStream()` with exponential reconnect (1s→2s→5s→10s cap) and jitter.
- Dispatch table routes events to store mutators; unknown topic/type logged at debug level only.
- Backpressure: if message queue length > 500, switch to degraded mode (pause processing, request resync via REST once drained).

### 21.3 Resync Protocol
If the client detects a gap (missed heartbeat > 30s or sequence jump), it issues parallel REST fetches for affected topics to rehydrate state, then resumes incremental updates.

Heartbeat Frame:
```json
{ "topic":"control", "type":"heartbeat", "ts":"2025-09-10T10:00:15Z" }
```
Timeout: 2 * heartbeat interval (default 15s).

### 21.4 Security / Future
- Phase 2: JWT passed in initial handshake, server validates and scopes topics.
- Optional signature per frame (HMAC) if events drive privileged visuals.

## 22. KG Visualization Component Draft
### 22.1 Component Responsibilities
`KgGraph` renders a sampled subset or summarized representation of the knowledge graph for exploratory inspection (UI-FR-018). It does not act as an editor. Priorities: fast mount, minimal bundle impact, graceful fallback.

### 22.2 Props Contract
```ts
interface KgGraphProps {
  summary: KnowledgeGraphSummary // must include degree_histogram & top_entities if available
  fetchFullGraph?: () => Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> // optional lazy detail
  height?: number
  theme?: 'light' | 'dark'
}
interface GraphNode { id: string; label: string; degree?: number; entityType?: string }
interface GraphEdge { id: string; source: string; target: string; kind?: string; weight?: number }
```

### 22.3 Rendering Modes
| Mode        | Trigger                           | Behavior                                                              |
|-------------|-----------------------------------|-----------------------------------------------------------------------|
| SummaryOnly | Default (no fetchFullGraph)       | Show degree histogram (SVG), top entities list, “Enable Graph” button |
| LazyGraph   | User clicks enable & flag true    | Dynamic import cytoscape, build elements from sampled nodes/edges     |
| Expanded    | fetchFullGraph resolves large set | Apply layout (cose-bilkent) and enable fit-on-select                  |
| Degraded    | Import or layout error            | Display textual fallback summary                                      |

### 22.4 Dynamic Import Pattern
```ts
const CytoscapeGraph = React.lazy(() => import(/* webpackChunkName: "kg-viz" */ './CytoscapeGraph'))
```
Suspense boundary fallback: skeleton + textual summary.

### 22.5 Layout & Styling
- Default layout: concentric for ≤ 200 nodes; switch to cose-bilkent for > 200 nodes (runtime import extension if needed).
- Node color ramp by degree percentile (quintiles). Edge opacity scaled by weight (if provided).
- Theming via CSS vars; dark mode toggles background and node stroke contrast.

### 22.6 Performance Considerations
- Cap initial rendered nodes to 500; if more present, show “Sampled 500 of N” pill.
- Use WebGL renderer only if available & node count > 800; fallback to canvas.
- Debounce resize & fit calls (100ms).

### 22.7 Accessibility
- ARIA live region announcing: "Graph loaded with X nodes and Y edges".
- Provide table toggle listing top entities for screen readers.

### 22.8 Error Cases
| Case                   | Handling                                                  |
|------------------------|-----------------------------------------------------------|
| Dynamic import failure | Record error, show fallback summary + retry button        |
| fetchFullGraph reject  | Show partial summary with notice "Full graph unavailable" |
| > max JSON size (2MB)  | Abort fetch, degrade to summary only                      |

### 22.9 Testing Hooks
- Data-testids: kg-summary-hist, kg-top-entities, kg-enable-btn, kg-graph-canvas.
- Mock dynamic import in unit tests to bypass real cytoscape.

### 22.10 Future Enhancements
- Subgraph filtering (entity type / degree range).
- Node detail side panel with originating chunk references.
- Snapshot export (PNG/SVG) via cytoscape exporter.

## 23. WebSocket Schema & Recovery
### 23.1 Envelope Schema (JSON Schema v2020-12 excerpt)
```json
{
  "$id": "https://example.com/schemas/ui-event-envelope.schema.json",
  "type": "object",
  "required": ["topic", "type", "ts", "seq", "data"],
  "properties": {
    "topic": {"type": "string"},
    "type": {"type": "string"},
    "ts": {"type": "string", "format": "date-time"},
    "seq": {"type": "integer", "minimum": 0},
    "trace_id": {"type": "string"},
    "schemaVersion": {"type": "string", "default": "v1"},
    "data": {"type": "object"}
  }
}
```
Forward compatibility: unknown properties ignored. Backward: removal requires deprecation window (≥1 release).

### 23.2 Sequence & Gap Detection
- `seq` monotonic per topic partition (topic-level ordering); client tracks last seq per topic.
- Gap policy: if `incoming.seq > last_seq + 1` → mark topic stale -> enqueue resync.

### 23.3 Recovery Algorithm (Pseudocode)
```
onEvent(e):
  if gap(e.topic, e.seq):
     markStale(e.topic)
  buffer(e)
  if staleCount() > 0 and !resyncInFlight:
     startResync()

startResync():
  for each staleTopic -> REST refetch latest snapshot
  merge snapshot, drop buffered older events
  clear stale marks
```

### 23.4 Close Codes (Planned)
| Code | Meaning                   | Client Action                           |
|------|---------------------------|-----------------------------------------|
| 4000 | Auth expired              | Reconnect (after token refresh Phase 2) |
| 4001 | Protocol version mismatch | Downgrade to polling + log error        |
| 4002 | Too many subscriptions    | Reduce topics set & retry               |
| 4003 | Rate limit                | Backoff (min 10s) then retry            |

### 23.5 Max Frame & Compression
- Max frame size: 64KB (client discards larger with error telemetry `ui.ws.frame_oversize`).
- Compression: optional permessage-deflate; disable if latency < 50ms RTT & payload < 1KB p95.

### 23.6 Heartbeat & Timeout
- Heartbeat event every 15s; if >30s since last heartbeat frame or any data event, trigger reconnect.
- Heartbeat drift metric: difference between expected and actual arrival time.

### 23.7 Fallback Escalation Flow
```
poll → attempt WS (success) → WS failure → exponential reconnect (1,2,5,10s) → after 3 failures stay on poll for 2 min → retry upgrade
```

## 24. KgGraph Testing & Performance Budget
### 24.1 Unit Test Matrix
| ID        | Scenario                          | Expectation                                     |
|-----------|-----------------------------------|-------------------------------------------------|
| KG-UT-001 | summary only (no fetchFullGraph)  | Renders histogram + entities, no dynamic import |
| KG-UT-002 | enable graph success              | Mode transitions summary→loading→graph          |
| KG-UT-003 | enable graph error                | Shows retry path + error message                |
| KG-UT-004 | node cap 600 supplied             | Renders 500 nodes message pill                  |
| KG-UT-005 | missing histogram                 | Displays fallback text                          |
| KG-UT-006 | lazy import failure (mock reject) | Degraded mode + retry                           |
| KG-UT-007 | a11y summary text present         | Contains screen reader region                   |

### 24.2 Integration / E2E
| ID         | Scenario            | Assertion                                          |
|------------|---------------------|----------------------------------------------------|
| KG-E2E-001 | flag off            | Container shows disabled message, no network fetch |
| KG-E2E-002 | flag on + enable    | Graph canvas present, chunk loaded                 |
| KG-E2E-003 | large dataset fetch | Sampling pill visible                              |

### 24.3 Performance Budgets (Initial)
| Metric                                   | Budget             | Measurement Method                 |
|------------------------------------------|--------------------|------------------------------------|
| Cytoscape chunk (gz)                     | ≤ 300KB            | build stats analyzer               |
| KgGraph wrapper chunk                    | ≤ 40KB             | build stats analyzer               |
| Summary render time (no graph)           | < 50ms main thread | performance.now delta test harness |
| Lazy graph first paint (≤500 nodes)      | < 1200ms           | E2E trace (Performance API)        |
| Memory increment after mount (500 nodes) | < 30MB             | Chrome heap snapshot diff          |

### 24.4 Regression Guard
- CI step parses Vite manifest sizes; fails if thresholds exceeded (allow 5% variance tolerance).
- Synthetic benchmark script mounts KgGraph 20 times to detect memory leaks (heap not growing >10% after GC).

## 25. Telemetry & Metrics Spec
### 25.1 Event Taxonomy
| Event Type       | Fields                              | Trigger                        |
|------------------|-------------------------------------|--------------------------------|
| ui.ws.connect    | { ts, attempt, success }            | WS open/close                  |
| ui.ws.gap        | { topic, last_seq, incoming_seq }   | Gap detected                   |
| ui.ws.resync     | { topics[], duration_ms }           | Resync completed               |
| ui.ws.downgrade  | { reason }                          | Switch to polling              |
| ui.kg.enable     | { node_goal, fetch_ms }             | User enables graph             |
| ui.kg.render     | { nodes, edges, mode, duration_ms } | After cytoscape initial layout |
| ui.flag.snapshot | { flags, schemaVersion }            | On boot                        |

### 25.2 Derived Metrics
- WS Uptime % = ws_connected_time / session_time.
- Reconnect Rate = reconnect_events / hour.
- KG Layout Time p95.
- Flag Fetch Failure Rate.

### 25.3 Log & Sampling Policy
- High-frequency events (progress.update) not individually logged; aggregated counters updated in memory and flushed every 60s.
- Error-class events always logged.

### 25.4 Privacy
- No PII in telemetry payloads; redact values matching secret patterns (/{api|token|key}/i).

## 26. ADR References (Planned)
Updated ADR mapping to repository ADR documents (see `docs/adr/`). Some earlier placeholders replaced by formal ADRs 001–006.

| ADR ID  | Title (Repository)                                      | Status | UI Design Cross-Refs     | Summary Impact                                     |
|---------|---------------------------------------------------------|--------|--------------------------|----------------------------------------------------|
| ADR-001 | Microservices Structure                                 | Draft  | §2.2 (service endpoints) | Confirms backend service boundaries consumed by UI |
| ADR-002 | Knowledge Graph Visualization Technology (Cytoscape.js) | Draft  | §3.2, §22                | Validates library & lazy load strategy (KG panel)  |
| ADR-003 | Subgraph Sampling Strategy                              | Draft  | §22, §27                 | Predictable sampled graph → stable UI caching      |
| ADR-004 | Manifest Integrity & Artifact Traceability              | Draft  | §3 (artifact fetch), §25 | Enables future integrity checks surfaced in UI     |
| ADR-005 | Telemetry Taxonomy & Naming Conventions                 | Draft  | §15, §25                 | Normalizes event keys & metrics naming             |
| ADR-006 | Event Schema Versioning Strategy                        | Draft  | §21, §23                 | Governs WS/event evolution & client validation     |

Superseded placeholders: prior “KG Visualization Library” row now formalized as ADR-002; “Graph Sampling Cap” incorporated into ADR-003 (cap + determinism). Feature flag merge strategy remains implicit within existing provider; may get its own ADR if divergence risk increases.

## 27. Subgraph API Draft (Exploratory)
Purpose: Provide on-demand focused graph slices to support future UI features (filtering, contextual expansion) without transferring entire KG.

### 27.1 Endpoint
`GET /kg/{kg_id}/subgraph`

### 27.2 Query Parameters
| Param          | Type       | Required                  | Description                      | Constraints                    |
|----------------|------------|---------------------------|----------------------------------|--------------------------------|
| center         | string     | yes (unless entity query) | Node ID or entity label seed     | must exist or 404              |
| entity         | string     | alternative to center     | Exact entity string lookup       | mutually exclusive with center |
| hop            | int        | optional                  | Undirected hop distance          | 1..3 (default 1)               |
| max_nodes      | int        | optional                  | Upper bound nodes returned       | 50..500 (default 200)          |
| relation_types | csv string | optional                  | Filter edges by relation kind    | validated against whitelist    |
| degree_min     | int        | optional                  | Minimum node degree              | >=0                            |
| degree_max     | int        | optional                  | Maximum node degree              | >= degree_min                  |
| summarize      | bool       | optional                  | If true, include aggregate stats | default true                   |
| version        | string     | optional                  | KG schema version                | fallback current               |

### 27.3 Response (schemaVersion v1)
```json
{
  "kg_id": "uuid",
  "center": "node-123",
  "hop": 2,
  "node_count": 180,
  "edge_count": 340,
  "truncated": true,
  "nodes": [
    { "id": "node-123", "label": "鋼板", "degree": 42, "entityType": "material" },
    { "id": "node-987", "label": "檢查", "degree": 12 }
  ],
  "edges": [
    { "id": "e-1", "source": "node-123", "target": "node-987", "kind": "action", "weight": 0.76 }
  ],
  "summary": {
    "degree_histogram": [5,12,40,22,7],
    "top_entities": [{"entity":"鋼板","degree":42},{"entity":"檢查","degree":12}],
    "avg_weight": 0.55,
    "density": 0.021
  },
  "schemaVersion": "v1"
}
```

### 27.4 Truncation & Sampling
- If resulting node set > `max_nodes`, apply deterministic sampling (hash(id) modulo) to ensure stable subsets across requests with same parameters.
- Set `truncated=true` when sampling applied; maintain original counts in `node_count` / `edge_count` reflecting the un-sampled subgraph.

### 27.5 Error Model
| HTTP | Code             | Example message                            | Notes                     |
|------|------------------|--------------------------------------------|---------------------------|
| 400  | VALIDATION_ERROR | "max_nodes out of range (min 50, max 500)" | parameter constraint      |
| 400  | MUTUAL_EXCLUSION | "center and entity are mutually exclusive" | choose one                |
| 404  | NOT_FOUND        | "center node not found"                    | center/entity not present |
| 410  | GONE             | "kg version deprecated"                    | version no longer served  |
| 429  | RATE_LIMIT       | "too many subgraph requests"               | protect backend           |

### 27.6 Caching & ETag
- Cache key includes (kg_id, center|entity, hop, filters, max_nodes, version).
- Recommend `Cache-Control: public, max-age=30, stale-while-revalidate=60`.
- Provide strong ETag over normalized (sorted nodes+edges) JSON excluding volatile fields (e.g., timestamps).

### 27.7 Rate Limiting (Guideline)
- Soft limit: 30 requests / minute / user for same kg_id.
- Hard burst: 5 requests within 5 seconds triggers 429 with Retry-After.

### 27.8 Security / Abuse Considerations
- Ensure filtered view does not leak redacted nodes (apply PII redaction before sampling).
- Deny hop > 3 to contain traversal explosion.
- Monitor average node/edge payload size; adjust default max_nodes if p95 payload > 1MB.

### 27.9 Future Extensions
- `direction=out|in|both` for directed edge traversal.
- Weighted radius expansion (stop when cumulative edge weight below threshold).
- Server-provided layout hints `{ positions: { nodeId: {x,y} } }` to reduce client layout cost.

### 27.10 Open Questions
- Should sampling be stratified by entityType to preserve minority categories?
- Include minimal provenance (e.g., top 2 source chunk IDs per node) without ballooning payload?
- Allow batching: POST /kg/subgraphs { requests:[...] } for multi-center queries?

---
End of document.

## 28. End-to-End Data Flow Perspective
This section integrates a consolidated data flow viewpoint aligning backend service phases, produced artifacts, events, idempotency anchors, UI consumption patterns, and requirement traceability (requirements.md, requirements.ui.md). It complements Section 3 (sequence) by adding lifecycle-wide lineage and operational guardrails.

### 28.1 Phase Overview Table
| Phase               | Primary Inputs                  | Transform Core                                     | Output Artifacts (Object Storage paths)                         | Emitted Events                            | Downstream Consumers    | Key Reqs                      |
|---------------------|---------------------------------|----------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------|-------------------------|-------------------------------|
| Ingestion           | km_id, version                  | Fetch, checksum, dedupe                            | documents/<km_id>/<version>/raw                                 | document.ingested                         | Processing UI           | FR ingest set, UI-FR-003/004  |
| Processing          | document_id                     | Extract → normalize → chunk → embed                | chunks/<document_id>/chunks.jsonl                               | document.processed / processing.completed | Testset, KG             | FR processing, UI-FR-008~012  |
| KG Build (opt)      | chunk ids + embeddings          | Entity/keyphrase extraction, relationship builders | kg/<kg_id>/graph.json, kg/<kg_id>/summary.json                  | kg.built                                  | Testset strategy, KG UI | FR KG (future), UI-FR-016~018 |
| Testset Gen         | chunks (+kg_id?) + config(seed) | Q/A synthesis, persona/scenario generation, dedupe | testsets/<id>/samples.jsonl, personas.json, scenarios.json      | testset.created                           | Evaluation              | FR-013~016, UI-FR-019~022     |
| Evaluation Run      | testset_id + rag profile        | RAG query, context capture, metrics calc           | runs/<run_id>/evaluation_items.json, kpis.json, thresholds.json | eval_runs.progress.update / run.completed | Insights, Reporting     | FR-017~022, UI-FR-023~026     |
| Insights Adapter    | evaluation artifacts            | Normalize IDs, aggregate, optional summary         | runs/<run_id>/export_summary.json (flagged)                     | adapter.exported                          | Insights Portal         | FR-038/039, UI-FR-027~029     |
| Reporting           | run artifacts + export summary  | HTML render, PDF generation                        | runs/<run_id>/report.html, report.pdf, run_meta.json            | report.completed                          | Reports UI, KM export   | FR-037~040, UI-FR-030~032     |
| KM Export (initial) | testset & kg summaries          | Minimal filter, redact sensitive                   | km_exports/testset_summary_v0.json, kg_summary_v0.json          | km.exported                               | KM system, KM UI        | FR-041/042, UI-FR-033~035     |
| Feature Flags       | static + service config         | Merge + snapshot                                   | (client memory)                                                 | (none)                                    | All lazy modules        | UI-NFR-006                    |
| Subgraph (draft)    | kg_id + center/entity           | Deterministic sampling + stats                     | ephemeral JSON response                                         | (future) subgraph.served                  | KG visualization        | Spec §27                      |

### 28.2 Artifact Lineage Chain
documents/raw → chunks.jsonl → (graph.json) → samples.jsonl → evaluation_items.json → kpis.json → export_summary.json (optional) → report.html/pdf + run_meta.json → km summaries (testset_summary_v0 / kg_summary_v0)

Traceability Goal: SMART objective #4 (≥95% sample-to-source linkage) supported by preserving: sample.question.source_chunk_ids[], evaluation_items[].chunk_refs[], graph node chunk_ids[].

### 28.3 Idempotency & Determinism Anchors
| Stage      | Deterministic Hash Basis                    | Purpose                              |
|------------|---------------------------------------------|--------------------------------------|
| Processing | (document_id + profile_hash)                | Avoid duplicate chunk/embedding work |
| KG Build   | (kg_build_config hash)                      | Rebuild decision vs reuse            |
| Testset    | (seed + normalized config + optional kg_id) | Reproduce identical sample set       |
| Evaluation | (testset_id + rag_profile_hash)             | Future metrics cache key             |
| KM Export  | (resource_type + source_run_id)             | Prevent redundant summary artifacts  |

### 28.4 UI Fetch Strategy Mapping
Polling cadences and downgrade paths consolidated (see §7 Chinese version for table). Reinforces performance NFRs and UI-FR-050 throttling requirement.

### 28.5 Event → UI Requirement Mapping
Minimal event inventory meets realtime UX without over-emission (WS bandwidth guard). See also envelope §23.

| UI-FR         | Event/API                                  | Purpose                |
|---------------|--------------------------------------------|------------------------|
| UI-FR-023/024 | GET /eval-runs + eval_runs.progress.update | Live progress          |
| UI-FR-030~032 | report.completed + GET /reports/:id        | Report list & fallback |
| UI-FR-033~035 | GET /exports?type=km_summary               | KM summary refresh     |
| UI-FR-016~018 | GET /kg/:id/summary + subgraph (draft)     | KG visualization       |
| UI-FR-049~051 | Throttled polling + WS latency tracking    | Performance SLA        |
| UI-FR-053~055 | Unified error envelope                     | Reliability surfacing  |

### 28.6 Risk & Guardrails (Data Flow)
| Risk                  | Layer             | Mitigation                               |
|-----------------------|-------------------|------------------------------------------|
| Excess polling load   | API gateway       | Global rate + client backoff             |
| Large KG payload      | KG service        | Summary + subgraph sampling (≤500 nodes) |
| Progress drift        | WS                | Sequence gap detection + REST resync     |
| Report link stale     | Reporting         | Atomic run_meta update + HTML fallback   |
| Summary inconsistency | KM export         | schema_version + (future) manifest hash  |
| Subgraph explosion    | Subgraph endpoint | hop ≤3 & deterministic sampling          |

### 28.7 Future Hooks
| Hook                        | Current         | Next Step                                   |
|-----------------------------|-----------------|---------------------------------------------|
| Manifest integrity          | Not implemented | Add manifest.json with sha256 per artifact  |
| Event schema registry       | Spec only       | JSON Schema validation & versioning         |
| Multi-center subgraph batch | Open question   | POST /kg/subgraphs exploration              |
| Persona/scenario deep drill | Basic counts    | Add drill-down lineage view                 |
| Metrics caching             | Planned         | Stage-level cache keyed by determinism hash |

### 28.8 Cross-Document References
requirements.md (SMART #4, FR-037~042), requirements.ui.md (UI-FR groupings), design.md (§4, §18), this file (§3, §23–27) – ensuring consistent data contracts & lineage narrative.

### 28.9 Implementation Notes
Documentation only; runtime unchanged. Future OpenAPI additions must version endpoints and keep WebSocket envelope stable (additive fields only) to preserve backwards compatibility.

