# UI Platform Requirements (Evaluation Workflow Console)

Version: 0.1 (Draft)
Status: Draft for review
Date: 2025-09-10


---
## 1. Purpose
Provide a unified web UI to operate, monitor, and analyze the multi‑stage RAG evaluation workflow (Document Ingestion → Processing → KG Build → Testset Generation → Evaluation → Insights/Reporting) replacing ad‑hoc CLI usage while supporting role‑specific perspectives.

## 2. Scope & Non‑Goals
In Scope:
- Trigger & monitor pipeline stages through service APIs (existing and forthcoming microservices).
- Role dashboards for engineering (pipeline ops) and analytical (evaluation quality) personas.
- Basic run artifact browsing & download.
- KM summary visibility (testset & KG) per FR‑041/042.
- Containerized deployment (dev docker‑compose; prod k8s ready).
Out of Scope (v1):
- Direct model inference UI playground (future enhancement).
- Full derivative resource management (deferred to DR‑001/DR‑002 lifecycle resolution).
- Multi‑tenant org segregation (future).

## 3. Personas
- KM Engineer (KME)
- Data Processing Engineer (DPE)
- Knowledge Graph Engineer (KGE)
- Testset Generation Engineer (TGE)
- RAG Evaluation Engineer (REE)
- QA / Evaluation Analyst (QA)
- Product / PM (PM)
- Platform / SRE (SRE)
- Security & Compliance (SEC)

## 4. EARS Requirement Syntax Key
- Ubiquitous: Always applicable behavior.
- Event‑driven: Triggered by an event or user action.
- State‑driven: Dependent on a persisted state.
- Optional / Feature‑flagged: Enabled only when feature flag on.
- Unwanted Behavior: Negative condition handling.

## 5. Functional Requirements (UI-FR)

### 5.1 Core Navigation & Session
UI-FR-001 (Ubiquitous) The system shall provide a global navigation segmented by lifecycle: Documents, Processing, KG, Testsets, Evaluations, Insights, Reports, Admin.
UI-FR-002 (State-driven) The system shall persist the last selected lifecycle tab in local storage and restore on reload.
UI-FR-003 (Ubiquitous) The system shall expose a session indicator showing active persona, locale, environment (dev/prod), and workflow ID (if in context).
UI-FR-004 (Event-driven) When a user switches persona, the system shall adapt visible dashboards and default filters without reloading the full page.
UI-FR-005 (Unwanted Behavior) If persona profile fails to load, the system shall fallback to a Safe Minimal profile with generic tables.

### 5.2 Authentication & Access (Phase 2 Ready)
UI-FR-006 (Ubiquitous) The system shall support optional auth mode; when disabled all endpoints are treated as public dev mode with a visible banner.
UI-FR-007 (Event-driven) When a token expires, the system shall prompt re-auth without losing unsaved filter state.
UI-FR-008 (Unwanted Behavior) If an API call returns 403, the system shall hide restricted action buttons and display a permission hint.

### 5.3 Document Ingestion View
UI-FR-009 (Ubiquitous) The system shall list ingested documents with columns: km_id, version, checksum, size, status, last_event_ts.
UI-FR-010 (Event-driven) When a user submits a new document reference (km_id, version), the system shall call ingestion-svc POST /documents and surface job tracking.
UI-FR-011 (State-driven) While a document is in status=processing, the row shall auto-refresh every ≤10s.
UI-FR-012 (Unwanted Behavior) If ingestion fails (status=error), the system shall expose retry action conditioned on permission.

### 5.4 Processing Jobs View
UI-FR-013 (Ubiquitous) The system shall display processing jobs with progress (% chunks) and embedding profile hash.
UI-FR-014 (Event-driven) When a document.ingested event is received (poll/WS), the system shall offer a one-click “Start Processing”.
UI-FR-015 (Unwanted Behavior) If a processing job exceeds SLA threshold, the system shall visually flag (warning) and show elapsed vs SLA.

### 5.5 Knowledge Graph Dashboard
UI-FR-016 (Ubiquitous) The system shall list KG builds with node_count, relationship_count, build_profile_hash, status.
UI-FR-017 (Event-driven) When kg.built event arrives, the system shall update summary metrics (nodes, relationships delta vs prior build).
UI-FR-018 (Optional / Feature-flagged) When KG visualization flag is enabled, the system shall render a lightweight graph summary (degree distribution & top entities).

### 5.6 Testset Generation Console
UI-FR-019 (Ubiquitous) The system shall provide a form to configure testset generation (method, max_samples, seed, persona profile).
UI-FR-020 (Event-driven) When a testset job is submitted, the system shall display queued → running → completed status timeline.
UI-FR-021 (Unwanted Behavior) If sample_count exceeds configured limit, the system shall mark over-limit condition pre-submit and block submission.
UI-FR-022 (State-driven) The console shall show reproducibility hash (config hash + seed) after job creation.

### 5.7 Evaluation Run Orchestration
UI-FR-023 (Ubiquitous) The system shall list evaluation runs with columns: run_id, testset_id, metrics_version, progress, verdict.
UI-FR-024 (Event-driven) When run progress updates (WS/poll), the system shall update progress bar and partial metrics preview.
UI-FR-025 (Event-driven) When a run completes, the system shall link to insights adapter outputs and reports if available.
UI-FR-026 (Unwanted Behavior) If metric computation fails for a sample, the system shall increment error_count and show a tooltip with last_error.

### 5.8 Insights & Monitoring Integration
UI-FR-027 (Ubiquitous) The system shall embed or link to existing Insights Portal views for per-run analysis reusing its component library where feasible.
UI-FR-028 (Event-driven) When a user selects a run in Evaluation list and opens Insights, the system shall pass run_id context and pre-load baseline metrics.
UI-FR-029 (Optional / Feature-flagged) When multi-run compare mode is toggled, the system shall permit selection of up to 5 runs and highlight KPI deltas.

### 5.9 Reporting & Artifact Access
UI-FR-030 (Ubiquitous) The system shall list generated reports (HTML/PDF) and allow in-browser preview (HTML) and download (PDF).
UI-FR-031 (Event-driven) When report generation starts, the system shall display status spinner until report.completed event.
UI-FR-032 (Unwanted Behavior) If report PDF link 404s, the system shall attempt HTML fallback and annotate degraded mode.

### 5.10 KM Export Summaries
UI-FR-033 (Ubiquitous) The system shall expose testset & KG KM summaries (FR-041/042) with schema_version and created_at.
UI-FR-034 (Event-driven) When a new export summary appears, the system shall diff counts vs previous and highlight increases.
UI-FR-035 (Unwanted Behavior) If KM summary validation fails JSON Schema, the system shall label the row invalid and suppress external publish action.

### 5.11 Operational Observability
UI-FR-036 (Ubiquitous) The system shall display service health (ingestion, processing, testset-gen, eval-runner, insights-adapter, reporting) via aggregated /health endpoints.
UI-FR-037 (Event-driven) When a service health transitions unhealthy, the system shall surface an alert banner with timestamp.
UI-FR-038 (State-driven) The system shall provide a diagnostics panel with latest trace_id samples and error rates.

### 5.12 Deployment & Environment Management
UI-FR-039 (Ubiquitous) The system shall provide a configuration screen showing current API base URLs and feature flags (read-only in prod).
UI-FR-040 (Event-driven) When the user edits dev environment endpoints, the system shall validate reachability (HEAD /health) before persisting.
UI-FR-041 (Unwanted Behavior) If a Docker compose service is unreachable locally, the system shall show remediation tips (container name, expected port).

### 5.13 Extensibility & Plugin Surface
UI-FR-042 (Ubiquitous) The system shall load UI extension modules (ESM) from a configurable directory at startup (dev mode) for new panels.
UI-FR-043 (Event-driven) When an extension fails to register, the system shall log structured error and quarantine the module (no crash).
UI-FR-044 (Optional / Feature-flagged) When experimental metric visualizers are enabled, the system shall sandbox them in an iframe/worker boundary.

### 5.14 Internationalization & Accessibility
UI-FR-045 (Ubiquitous) The system shall support en-US and zh-TW locales for all UI strings.
UI-FR-046 (Event-driven) When locale changes, the system shall re-render visible text without full reload.
UI-FR-047 (Ubiquitous) The system shall meet WCAG 2.1 AA contrast ratios for text and interactive elements.
UI-FR-048 (Unwanted Behavior) If a translation key is missing, the system shall log it and display the key identifier as fallback.

### 5.15 Performance & Responsiveness
UI-FR-049 (Ubiquitous) The system shall load initial dashboard (Documents tab) ≤2s with ≤50 documents.
UI-FR-050 (State-driven) While polling lists, the system shall batch refreshes to no more than 1 request / 5s per list.
UI-FR-051 (Ubiquitous) The system shall render evaluation run list updates within 300ms of receiving progress events.
UI-FR-052 (Unwanted Behavior) If a single API response >2MB, the system shall warn and suggest enabling server-side pagination.

### 5.16 Reliability & Error Handling
UI-FR-053 (Ubiquitous) The system shall surface a unified error drawer capturing recent API errors (code, service, trace_id, ts).
UI-FR-054 (Event-driven) When network connectivity is lost, the system shall enter offline mode and queue permissible POST actions (if idempotent) until connectivity resumes.
UI-FR-055 (Unwanted Behavior) If a queued action exceeds max retry window (configurable), the system shall discard and notify user.

### 5.17 Security & Privacy
UI-FR-056 (Ubiquitous) The system shall redact secret values (API keys, tokens) visually after entry except last 4 chars.
UI-FR-057 (Event-driven) When a user copies a secret, the system shall log an audit event with user id and masked secret hash.
UI-FR-058 (Unwanted Behavior) If a response payload contains PII flagged fields, the system shall mask them according to policy.

### 5.18 Traceability & Audit
UI-FR-059 (Ubiquitous) The system shall capture audit events for create/update actions (who, what, when, resource_ref, trace_id).
UI-FR-060 (Event-driven) When viewing a run, the system shall show lineage chain (document → chunk → test sample → evaluation item) with hyperlink anchors.
UI-FR-061 (Unwanted Behavior) If lineage resolution fails for a node, the system shall mark the segment unresolved and continue rendering adjacent links.

### 5.19 CLI Parity & Transition
UI-FR-062 (Ubiquitous) The system shall provide a “Run CLI Equivalent” view showing the corresponding legacy CLI command for each submitted job.
UI-FR-063 (Event-driven) When a new job is created, the system shall generate the CLI command snippet (python3 run_pipeline.py ... ) and allow copy.
UI-FR-064 (Unwanted Behavior) If an option has no CLI equivalent, the system shall annotate it as (UI-only) in the snippet.

### 5.20 Future Derivatives Preparation
UI-FR-065 (Ubiquitous) The system shall reserve a Derivatives tab placeholder showing deferred resource types (chunk_index, evaluation_run_metrics, testset_schema) and status = deferred.
UI-FR-066 (Event-driven) When derivative governance (DR-001/DR-002) closes, the system shall enable creation/listing without structural UI overhaul.
UI-FR-067 (Unwanted Behavior) If a derivative record lacks mandatory draft schema fields, the system shall highlight and exclude from export actions.

## 6. Non-Functional UI Requirements (UI-NFR)
UI-NFR-001 Availability: Core navigation reachable ≥99% (underlying APIs assumed available) in production.
UI-NFR-002 Performance: P95 interaction latency < 500ms for tab switches (cached data).
UI-NFR-003 Internationalization: All new text keys localized before merge (CI l10n check).
UI-NFR-004 Accessibility: 0 critical axe-core violations in CI pipeline.
UI-NFR-005 Observability: UI emits structured logs (frontend logger) with trace_id correlation for ≥90% API calls.
UI-NFR-006 Bundle Size: Initial JS bundle ≤ 1.2MB gzipped (excluding map) for baseline persona view.
UI-NFR-007 Progressive Enhancement: Works with JS disabled for read-only static status pages (Documents minimal table + message).

## 7. Open Questions
- Should Insights Portal be embedded (iframe / micro-frontend) or merged codebase? (Initial decision: integrate as a module within existing portal repo, reuse analytics components.)
- What minimal graph visualization library (Cytoscape vs d3 force-lite) for KG summary? (Resolved: Cytoscape.js selected – see design Section 20 for rationale & lazy load strategy.)
- Do we expose direct WebSocket endpoints per service or multiplex through gateway? (Leaning gateway for auth consistency.)
- Scope of offline queued actions (restrict to idempotent create?).
- Derivatives tab activation criteria after DR-001/DR-002 resolution process.

## 8. Integration Decision (Initial)
Decision: Extend existing Insights Portal (monorepo module) rather than building a separate UI. Rationale:
1. Shared component & metrics rendering reduces duplication.
2. Existing persona & threshold frameworks accelerate delivery.
3. Single deployment artifact lowers operational overhead.
4. Future derivative artifacts can reuse current normalization pipeline.
Risk: Portal codebase grows in complexity; Mitigation: Introduce domain-based routing + lazy loaded bundles per lifecycle area.

## 9. Traceability
Mapping to existing platform requirements: UI-FR-033/034/035 ↔ FR-041/042 (KM summaries); UI-FR-060 ↔ global traceability objective (≥95% coverage); UI-FR-062/063 aids CLI transition plan.

---

