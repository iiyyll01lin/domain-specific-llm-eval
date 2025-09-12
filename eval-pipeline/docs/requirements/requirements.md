# RAG Evaluation Platform Microservices Transformation Requirements

Version: 0.1 (Draft)  
Status: Draft for review  
Date: 2025-09-09  
Owner: Platform Engineering  

---
## 1. Vision
Provide a modular, API-driven, containerized RAG Evaluation Platform where each life‑cycle stage (Document Ingestion → Processing → Testset Generation → RAG Evaluation → Analytics & Insights → Reporting) is exposed as independently deployable microservices with consistent contracts, observable behavior, and pluggable evaluation strategies, seamlessly integrated with a Knowledge Management (KM) REST API and the existing Insights Portal UI.

## 2. Objectives (SMART)
1. Replace single CLI entry (`run_pipeline.py`) with ≥6 bounded-context services within 3 iterations.  
2. Support end-to-end workflow orchestration through asynchronous job APIs (submit → status → result) under < 2s average request latency for control plane calls.  
3. Achieve horizontal scalability (≥ 3 replicas) for stateless services without code change.  
4. Provide traceability: Each test sample result links back to original document chunk and evaluation rules (≥ 95% coverage).  
5. Deliver bi-lingual (EN/zh-TW) API documentation & requirements.  

## 3. Stakeholders
- Platform Engineer: builds & maintains services.  
- Data / ML Engineer: integrates custom LLMs, embeddings, evaluation metrics.  
- QA / Evaluation Analyst: triggers evaluations and inspects results.  
- Product Manager: monitors KPI trends via Insights Portal.  
- Security / Compliance: audits PII handling, access control, retention.  
- External Integrators: push documents from KM system; consume evaluation events.  

## 4. Glossary
- KM API: External Knowledge Management REST endpoint providing document metadata & content retrieval.  
- Testset: Structured Q/A samples + contexts + metadata in normalized schema.  
- Evaluation Run: Immutable execution producing metrics per sample + aggregated KPIs.  
- Persona: Target user profile influencing scenario generation.  
- Scenario: Composite evaluation context (persona + query intent + constraints).  
- Insights Portal: Existing React/Vite analytics frontend consuming normalized run artifacts.  
- Job Orchestration: Asynchronous workflow managing multi-stage processing.  

## 5. Service Decomposition (Initial)
| Service        | Responsibility                                                           | Key Endpoints (REST)                      | Data Store                 | Scaling   | Notes                                |
|----------------|--------------------------------------------------------------------------|-------------------------------------------|----------------------------|-----------|--------------------------------------|
| gateway-api    | Unified ingress, auth, routing, OpenAPI aggregation                      | /v1/* (reverse proxy)                     | n/a                        | stateless | Optional first iteration (can defer) |
| ingestion-svc  | Accept document references, fetch from KM API, persist raw & checksum    | POST /documents  GET /documents/:id       | Object store + metadata DB | CPU / I/O | Dedup & versioning                   |
| processing-svc | Chunking, text normalization, metadata enrichment (language, embeddings) | POST /process-jobs  GET /process-jobs/:id | Vector store + DB          | CPU / GPU | Idempotent chunk pipeline            |
| testset-gen-svc | Generate questions, scenarios, personas; supports methods: configurable|ragas|hybrid | POST /testset-jobs  GET /testset-jobs/:id  GET /templates | DB + object store | GPU/LLM bound | Pluggable strategies |
| kg-builder-svc | Knowledge graph creation & relationship enrichment | POST /kg-jobs  GET /kg/:id | Graph DB / doc store | CPU / embedding | Optional split from processing |
| eval-runner-svc | Execute RAG queries against target system, compute metrics (contextual, RAGAS, semantic, human-feedback flags) | POST /eval-runs  GET /eval-runs/:id  WS /eval-runs/:id/stream | DB + metrics store | CPU / network | Extensible metric registry |
| insights-adapter-svc | Transform run outputs into portal-compatible artifacts (CSV, JSON, XLSX meta) & push or expose | POST /exports  GET /exports/:id  GET /runs/:id/insights | Object store | stateless | Bridges to Insights Portal |
| reporting-svc | PDF/HTML executive + technical reports generation | POST /reports  GET /reports/:id | Object store | CPU | Puppeteer / headless Chrome |
| orchestrator-svc | DAG/job state machine across services; retry, compensation, SLA tracking | POST /workflows  GET /workflows/:id | DB (workflow) | stateless | Could use temporal / arq / celery |
| authz-svc (future) | Fine-grained RBAC, API tokens, scoped secrets | POST /tokens | DB (secrets) | stateless | Phase 2 |

## 6. High-Level Flow
1. Ingestion: Client POST /documents with KM reference (km_id, version).  
2. Processing job creates chunks + embeddings → stored & emits event.  
3. Optional KG build uses processed chunks.  
4. Testset generation job produces test samples + personas + scenarios.  
5. Evaluation runner queries external RAG system & computes metrics.  
6. Insights adapter normalizes & pushes artifacts consumable by Insights Portal schemas (`EvaluationItem`, KPI aggregation).  
7. Reporting generates branded PDF/HTML & notifies client.  
8. Orchestrator tracks workflow state & error retries.  

## 7. External Integrations
### 7.1 KM REST API (Placeholder)
- Base: https://km.example.com/api  
- Auth: Bearer token (header Authorization).  
- GET /documents/:id → { id, title, version, mime_type, size_bytes, checksum, updated_at }
- GET /documents/:id/content → stream (or presigned URL)  
- GET /documents/:id/metadata → domain-specific kv pairs.  
- Error model: { error_code, message, retryable }  

### 7.2 RAG Target System
- Configurable endpoint: POST /query { query, top_k, persona, scenario_id, trace:bool } → { answer, contexts:[{id,text,score}], latency_ms, model, tokens }  
- SLA: < 5s p95 latency for top_k ≤ 10.  

### 7.3 Insights Portal
- Consumes exported run artifacts:  
  - evaluation_items.json (array of `EvaluationItem`)  
  - kpis.json (aggregated metrics)  
  - thresholds.json (optional profile)  
  - personas.json (optional)  
  - run_meta.json (workflow + version info)  
- Export adapter ensures alignment with `schemas.ts` normalizers (id stability, metric keys).  

### 7.4 Derivatives API (Placeholder / Deferred)
Purpose (Deferred – DR-001 / DR-002): Provide a unified interface to persist and retrieve derivative artifacts produced from evaluation runs (e.g., `chunk_index`, `kg_summary`, `evaluation_run_metrics`, `testset_schema`). Enablement is deferred until derivative governance & retention are finalized.

Proposed Endpoints (all deferred):
| Method | Endpoint                  | Description                                                      | Notes                                              |
|--------|---------------------------|------------------------------------------------------------------|----------------------------------------------------|
| POST   | /derivatives              | Create derivative resource (metadata + optional payload pointer) | Idempotent by (resource_type, source_run_id, hash) |
| GET    | /derivatives/:id          | Retrieve derivative metadata & access links                      | 404 if not found                                   |
| GET    | /runs/:run_id/derivatives | List derivatives for a run                                       | Filter by resource_type                            |
| DELETE | /derivatives/:id          | (Deferred) Soft-delete (tombstone) per retention policy          | Requires role=admin                                |

OpenAPI YAML Placeholder (Phase ≥3):
```
paths:
    /derivatives:
        post:
            summary: Create derivative (Deferred)
            operationId: createDerivative
            tags: [derivatives]
            requestBody:
                required: true
                content:
                    application/json:
                        schema:
                            $ref: '#/components/schemas/DerivativeCreateRequest'
            responses:
                '202': { description: Accepted }
    /derivatives/{id}:
        get:
            summary: Get derivative (Deferred)
            operationId: getDerivative
            tags: [derivatives]
            parameters:
                - name: id
                    in: path
                    required: true
                    schema: { type: string, format: uuid }
            responses:
                '200': { description: OK }
                '404': { description: Not Found }
    /runs/{run_id}/derivatives:
        get:
            summary: List derivatives by run (Deferred)
            operationId: listRunDerivatives
            tags: [derivatives]
            parameters:
                - name: run_id
                    in: path
                    required: true
                    schema: { type: string, format: uuid }
                - name: resource_type
                    in: query
                    required: false
                    schema: { type: string }
            responses:
                '200': { description: OK }
```

Notes:
- Security: Will require authz-svc (scoped token) & RBAC roles (admin for deletion).
- Storage: Object store (binary/payload) + relational/kv index (metadata & search).
- Retention: Governed by policy (see Section 14 Open Questions – DR-001).
- Partial resolution of DR-002 will appear in Section 8.8 (draft schema) without locking final enums.


## 8. Data Contracts (Core Schemas)
### 8.1 Document (Stored)
```
{
  "id": "uuid",
  "km_id": "string",
  "version": "string",
  "title": "string",
  "source_uri": "string",
  "mime_type": "application/pdf",
  "ingested_at": "ts",
  "checksum": "sha256",
  "size_bytes": 12345,
  "status": "active|archived",
  "metadata": {"language":"en","domain":"redfish"}
}
```
### 8.2 Chunk
```
{
  "id":"uuid",
  "document_id":"uuid",
  "order":0,
  "text":"...",
  "embedding_vector_id":"uuid",
  "token_count":512,
  "metadata": {"section":"3.1","language":"en"}
}
```
### 8.3 Test Sample
```
{
  "id":"uuid",
  "question":"string",
  "reference_answer":"string",
  "reference_context_ids":["chunk-id"],
  "keywords":["..."],
  "persona_id":"persona-uuid",
  "scenario_id":"scenario-uuid",
  "generation_method":"configurable|ragas|hybrid",
  "difficulty":"simple|multi_context|reasoning|complex",
  "metadata": {"seed":123}
}
```
### 8.4 Evaluation Result Item (Portal Compatible Subset)
```
{
  "id":"sample-id",
  "user_input":"question",
  "rag_answer":"string",
  "reference":"reference_answer",
  "reference_contexts":["ctx1","ctx2"],
  "rag_contexts":["retr_ctx1","retr_ctx2"],
  "metrics":{
    "Faithfulness":0.82,
    "AnswerRelevancy":0.87,
    "ContextPrecision":0.79,
    "ContextRecall":0.81,
    "AnswerSimilarity":0.80,
    "ContextualKeywordMean":0.84
  },
  "latencyMs":1340,
  "extra": {"model":"llama2-7b","tokens":1234}
}
```
### 8.5 KPI Aggregation
```
{
  "run_id":"uuid",
  "metrics": {"Faithfulness":0.82, "ContextPrecision":0.79, ...},
  "counts": {"total":100, "needs_human":7},
  "latency": {"avg":1200, "p50":900, "p90":2200, "p99":4000},
  "threshold_profile_id":"default-v1"
}
```

### 8.6 Run Meta (Extended)
```
{
    "run_id":"uuid",
    "created_at":"ts",
    "pipeline_version":"string",
    "metrics_version":"v1",
    "report_html_url":"https://object-store/runs/<id>/report.html",
    "report_pdf_url":"https://object-store/runs/<id>/report.pdf",  // optional
    "export_profile":"default",
    "extras": {}
}
```

### 8.7 Export Summary (Optional Artifacts)
```
{
    "run_id":"uuid",
    "kg_summary":{"node_count":120,"relationship_count":340},      // present if KG used
    "persona_stats":{"persona_count":5,"scenario_count":12},       // present if personas generated
    "feature_flags":{"kg_summary_export":true,"persona_stats":true}
}
```

### 8.8 Derivative Resource (Deferred Draft)
Status: Draft (Partial resolution of DR-002; final contract deferred to design phase). This schema captures minimal common metadata for future derivative artifacts produced by evaluation runs. Enumeration values & retention mechanics intentionally incomplete.

Example Instance:
```
{
    "derivative_id": "uuid",
    "source_run_id": "uuid",
    "resource_type": "kg_summary",             // candidate enum; NOT final
    "version": "v0",
    "created_at": "ts",
    "hash": "sha256",
    "pii_classification": "none|low|moderate|high",
    "content_uri": "s3://bucket/runs/<run>/derivatives/kg_summary.json",
    "size_bytes": 2048,
    "status": "available|expired|tombstoned",
    "metadata": {
        "node_count": 120,
        "relationship_count": 340
    }
}
```

JSON Schema (Draft):
```
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.com/schemas/derivative-resource.schema.json",
    "title": "DerivativeResource",
    "type": "object",
    "required": [
        "derivative_id",
        "source_run_id",
        "resource_type",
        "created_at",
        "hash",
        "content_uri",
        "status"
    ],
    "properties": {
        "derivative_id": { "type": "string", "format": "uuid" },
        "source_run_id": { "type": "string", "format": "uuid" },
        "resource_type": { "type": "string", "description": "Provisional enumerated type (e.g., chunk_index, kg_summary, evaluation_run_metrics, testset_schema)" },
        "version": { "type": "string", "default": "v0" },
        "created_at": { "type": "string", "format": "date-time" },
        "hash": { "type": "string", "description": "SHA-256 (or future) integrity hash" },
        "pii_classification": { "type": "string", "description": "Risk tier; classification taxonomy TBD" },
        "content_uri": { "type": "string", "format": "uri" },
        "size_bytes": { "type": "integer", "minimum": 0 },
        "status": { "type": "string", "description": "Lifecycle state (available|expired|tombstoned) - open set" },
        "metadata": { "type": "object", "additionalProperties": true }
    },
    "additionalProperties": false
}
```

Non-Goals (This Draft):
- Final enum closure for resource_type.
- Retention & purge policy (depends on DR-001 outcome).
- Cross-resource referencing (e.g., linking derivative-to-derivative).

Planned Validation Rules (Design Phase):
- (Rule) resource_type + source_run_id + hash must be unique.
- (Rule) If status=tombstoned then content_uri must not be publicly resolvable.
- (Rule) pii_classification != 'none' ⇒ access requires elevated scope.


## 9. Functional Requirements (EARS)
Legend:  
- Ubiquitous form: "The <system/component> shall ..."  
- Event-driven form: "When <condition>, the <component> shall ..."  
- Optional feature: "Where <feature flag> is enabled, the <component> shall ..."  
- Unwanted behavior: "The <component> shall not ..."  

### 9.1 Ingestion Service
1. (UBI) The ingestion-svc shall accept document creation requests referencing an external KM document id.  
   Acceptance: POST /documents with km_id returns 202 + {document_id}.  
2. (EVENT) When a document is ingested, the ingestion-svc shall compute and store a SHA-256 checksum.  
   Acceptance: GET /documents/:id includes checksum matching recomputation.  
3. (EVENT) When a duplicate checksum is detected for same km_id+version, the ingestion-svc shall not create a new record and shall return existing document id.  
   Acceptance: Second POST returns 200 with existing id.  
4. (UBI) The ingestion-svc shall fetch content via KM API streaming endpoint.  
   Acceptance: Simulated KM mock responds; content length stored equals source size_bytes.  
5. (EVENT) When content retrieval fails with retryable error_code, the ingestion-svc shall retry with exponential backoff (≥3 attempts).  
   Acceptance: Inject failure; logs show 3 retries then 5xx.  

### 9.2 Processing Service
6. (UBI) The processing-svc shall create a process job referencing a document_id and chunking profile.  
   Acceptance: POST /process-jobs returns job_id and initial status=queued.  
7. (EVENT) When a process job starts, the processing-svc shall extract text, detect language, and split into chunks respecting configured token limits.  
   Acceptance: For sample PDF, number of chunks * average token_count within ±10% target.  
8. (EVENT) When embedding generation is configured, the processing-svc shall produce embeddings stored in vector store with reference to chunk ids.  
   Acceptance: GET /process-jobs/:id shows embedding_count == chunk_count.  
9. (EVENT) When an error occurs mid-pipeline, the processing-svc shall mark job status=failed with failure_reason.  
   Acceptance: Force error; status becomes failed; reason persisted.  

### 9.3 Knowledge Graph Service
10. (UBI) The kg-builder-svc shall create a KG job referencing a list of document_ids or chunk set.  
    Acceptance: POST /kg-jobs returns kg_id and status=building.  
11. (EVENT) When nodes are created, the kg-builder-svc shall derive entities and keyphrases using hybrid extraction fallback.  
    Acceptance: >=1 entity list per node persisted.  
12. (EVENT) When RAGAS relationship builders are available, the kg-builder-svc shall compute similarity relationships; otherwise fallback custom heuristic.  
    Acceptance: For test dataset, at least one relationship per ≥50% nodes.  

### 9.4 Testset Generation Service
13. (UBI) The testset-gen-svc shall support generation_method configurable|ragas|hybrid per request.  
    Acceptance: POST /testset-jobs with each method yields artefacts tagged accordingly.  
14. (EVENT) When persona_generation is enabled, the testset-gen-svc shall output at least one persona and scenario per run.  
    Acceptance: Output contains personas.json and scenarios.json non-empty.  
15. (EVENT) When max_total_samples is exceeded by requested samples_per_document, the testset-gen-svc shall cap generation without failing.  
    Acceptance: Response shows total_samples == max_total_samples.  
16. (UNWANTED) The testset-gen-svc shall not produce duplicate question text within the same run.  
    Acceptance: All question strings unique set size == count.  

### 9.5 Evaluation Runner Service
17. (UBI) The eval-runner-svc shall execute evaluation runs referencing testset_id and target rag_system profile.  
    Acceptance: POST /eval-runs returns run_id; status progresses queued→running→completed.  
18. (EVENT) When querying the rag_system returns contexts, the eval-runner-svc shall compute context precision & recall using reference_contexts vs rag_contexts.  
    Acceptance: Metrics present in item metrics map.  
19. (EVENT) When RAGAS scoring is enabled, the eval-runner-svc shall compute Faithfulness and AnswerRelevancy.  
    Acceptance: Metrics contain both > 0 values.  
20. (EVENT) When keyword evaluation is enabled, the eval-runner-svc shall compute ContextualKeywordMean across weighted keywords.  
    Acceptance: Aggregated KPI exists in kpis.json.  
21. (EVENT) When uncertainty criteria satisfied (score range crossing adaptive band), the eval-runner-svc shall flag sample needs_human=true.  
    Acceptance: counts.needs_human > 0 for mixed-quality testset.  
22. (UNWANTED) The eval-runner-svc shall not mutate original testset samples.  
    Acceptance: Pre/post hash of testset unchanged.  

### 9.6 Insights Adapter Service
23. (UBI) The insights-adapter-svc shall normalize evaluation result items to match `normalizeItem` logic.  
    Acceptance: Fields align; derived metrics recognized by portal.  
24. (EVENT) When an eval run completes, the insights-adapter-svc shall publish artifacts set (evaluation_items.json, kpis.json, thresholds.json?).  
    Acceptance: Presence verified in object store.  
25. (EVENT) When export format=csv|xlsx requested, the insights-adapter-svc shall generate file using standardized exporter manifest including meta.  
    Acceptance: File headers contain runId & timestamp comments.  

### 9.7 Reporting Service
26. (UBI) The reporting-svc shall generate executive and technical HTML reports for completed run_id.  
    Acceptance: HTML includes KPI table and top N insights.  
27. (EVENT) When pdf=true, the reporting-svc shall convert executive summary to PDF via headless rendering.  
    Acceptance: Stored PDF size > 5KB and < 15MB.  

### 9.8 Orchestrator Service
28. (UBI) The orchestrator-svc shall allow workflow creation specifying ordered stage list.  
    Acceptance: POST /workflows returns workflow_id with stages array.  
29. (EVENT) When a stage fails with retryable flag, the orchestrator-svc shall attempt automatic retry (≥2) before marking failed.  
    Acceptance: Logs show retries, final status.  
30. (EVENT) When all stages succeed, the orchestrator-svc shall emit workflow_completed event.  
    Acceptance: Event log captured.  

### 9.9 Cross-Cutting
31. (UBI) Each service shall expose /healthz (liveness) and /readyz (readiness).  
32. (UBI) Each service shall emit structured JSON logs with trace_id & job_id correlation fields.  
33. (UBI) Each async job shall provide progress percentage (0-100).  
34. (UBI) Each API shall be documented via OpenAPI 3.1.  
35. (UNWANTED) Services shall not block event loop longer than configured max_request_time (default 30s).  
36. (EVENT) When configuration changes (hot-reload enabled), services shall reload without restart.  

### 9.10 Reporting & Portal Integration (New)
37. (EVENT) When a report is generated, the reporting-svc shall append report_html_url and (if PDF exists) report_pdf_url to run_meta.json.  
    Acceptance: run_meta.json contains valid URLs returning HTTP 200.  
38. (UBI / Optional) Where kg_summary_export is enabled, the insights-adapter-svc shall emit export_summary.json containing kg_summary node_count & relationship_count.  
    Acceptance: For a run with KG, counts > 0 and match KG store.  
39. (UBI / Optional) Where persona_stats_export is enabled, the insights-adapter-svc shall emit persona_stats in export_summary.json.  
    Acceptance: persona_count equals personas.json length; scenario_count equals scenarios.json length.  
40. (UBI) The platform shall expose a report download reference so the Insights Portal can render a download button sourced from run_meta.report_pdf_url (or fallback to HTML).  
    Acceptance: Portal receives run_meta with at least one report_*_url; link returns file >5KB.  

### 9.11 KM Storage (Initial Scope) (New)
41. (EVENT / Phase 1.5) When a testset is finalized (status=completed), the testset-gen-svc shall publish a KM export payload (testset_schema v0) to a staging topic or queue for KM ingestion containing: testset_id, sample_count, persona_count, scenario_count, generation_method, schema_version.  
    Acceptance: KM staging message visible; payload validates against draft schema (no full sample texts required yet).  
42. (EVENT / Phase 1.5) When a knowledge graph build completes, the kg-builder-svc shall produce a KM export summary with: kg_id, node_count, relationship_count, source_document_ids[], build_profile_hash.  
    Acceptance: Summary posted to KM staging; counts match internal graph.json.  
Deferred: Full derivative uploads (evaluation_run_metrics, chunk_index) remain blocked by DR-001 / DR-002 decisions.  

### 9.12 UI Integration (New)
Scope:
The unified operational UI (see `requirements.ui.md` & `requirements.ui.zh.md`) extends the existing Insights Portal rather than creating a standalone application. It enables role-based monitoring and control across lifecycle stages (Documents → Processing → KG → Testsets → Evaluations → Insights → Reports) and exposes KM export summaries (FR-041/042) plus future derivative placeholders.

Cross-Reference Mapping (Selected):
- UI-FR-009..012 align with Ingestion & Processing visibility (FR-001..009).
- UI-FR-016..018 surface KG metrics & relationships (FR-010..012).
- UI-FR-019..022 drive testset generation configuration (FR-013..016).
- UI-FR-023..026 monitor evaluation runs & metrics (FR-017..022, 37..40).
- UI-FR-030..032 map to reporting artifacts (FR-026..027, 37..40).
- UI-FR-033..035 display KM summaries (FR-041..042).
- UI-FR-060 supports traceability objective (Objective #4 ≥95%).
- UI-FR-062..064 provide CLI transition aids for migration strategy (Section 13).
- UI-FR-065..067 reserve derivative space pending DR-001 / DR-002 resolutions.

Non-Functional Alignment:
- UI-NFR-001 complements platform availability targets.
- UI-NFR-005 enforces trace_id propagation visibility (Observability Section 12).
- UI-NFR-006 ensures frontend bundle constraint supporting performance objectives.

Open Questions (UI Scope) summarized in UI requirements remain tracked separately; any closure impacting backend contracts must update this section.

## 10. Non-Functional Requirements
| Category             | Requirement (EARS style)                                                                                             | Acceptance                |
|----------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------|
| Performance          | The eval-runner-svc shall process ≥ 10 samples/sec (simple queries) on reference hardware.                           | Benchmark run log.        |
| Scalability          | The platform shall scale horizontally with linear throughput improvement ±25%.                                       | Load test graph.          |
| Reliability          | The orchestrator-svc shall persist workflow state ensuring restart recovery with ≤1 step duplication.                | Chaos test.               |
| Availability         | Each core service shall maintain ≥ 99% monthly uptime (excluding maintenance).                                       | Uptime dashboard.         |
| Security             | Where auth is enabled, services shall enforce token validation and reject unauthorized (401/403).                    | Auth test suite.          |
| Observability        | Each service shall export Prometheus metrics at /metrics including job_duration_seconds histogram.                   | Scrape validation.        |
| Traceability         | The platform shall propagate trace_id across all logs for a workflow end-to-end.                                     | Log sampling.             |
| Maintainability      | Each service shall reach ≥ 80% statement coverage on critical modules.                                               | CI report.                |
| Internationalization | The insights-adapter-svc shall support locale=en, zh-TW outputs for report metadata.                                 | Sample artifact diff.     |
| Extensibility        | The testset-gen-svc shall allow registering new generation strategy via entrypoint plugin without code modification. | Dynamic plugin load test. |
| Data Retention       | The ingestion-svc shall support configurable retention_days; expired documents shall be purged daily.                | Simulated purge job.      |
| Compliance           | Where PII is detected (regex rules), services shall mask values in logs.                                             | Log inspection.           |

## 11. Security & Privacy
- Token-based auth (future): Short-lived service tokens, optional mTLS.  
- Secrets: Mounted via runtime secrets store (K8s Secret / Vault), never logged.  
- PII Detection: Regex / model-based; masking strategy: replace middle chars with ***.  
- RBAC (Phase 2): Roles = viewer, runner, admin.  

## 12. Observability
- Logging: JSON lines {ts, level, svc, trace_id, job_id, msg}.  
- Metrics: Prometheus counters & histograms (request_total, request_duration_seconds, job_duration_seconds, failures_total).  
- Tracing: OpenTelemetry exporting OTLP to collector.  
- Dashboards: Grafana templates for per-stage throughput & error rate.  

## 13. Migration Strategy
Phase 0: Stabilize existing CLI; freeze feature set.  
Phase 1: Extract ingestion + processing + evaluation as services; orchestrator minimal.  
Phase 2: Add KG & testset-gen services with plugin strategy; integrate Insights Portal artifacts.  
Phase 3: Add reporting & gateway; harden security; add RBAC.  
Phase 4: Optimize performance (batching, streaming, vector index tuning).  
Fallback: CLI wrapper calls service APIs for continuity.  

### 13.1 Milestones (Indicative)
| Phase          | Milestone Focus         | Key Deliverables                                                                                     | Exit Criteria                                                     |
|----------------|-------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 0              | Stabilization           | CLI hardening, logging baseline                                                                      | No P1 defects in legacy run                                       |
| 1              | Core Extraction         | ingestion-svc, processing-svc, eval-runner-svc (FR 1-9,17-22), minimal orchestrator                  | End-to-end run via APIs produces evaluation_items.json            |
| 2              | Enrichment              | testset-gen-svc, kg-builder-svc, insights-adapter artifacts (FR 10-16,23-25,37-39 partial)           | Portal consumes normalized artifacts without manual patching      |
| 3              | Presentation & Security | reporting-svc, gateway-api, initial authz-svc (RBAC roles), report URLs in run_meta (FR 26-27,37,40) | Executive HTML/PDF accessible & auth enforced on protected routes |
| 3.5 (Deferred) | Derivatives Foundation  | Derivatives OpenAPI placeholder active (read-only list), internal prototype of /derivatives create   | Successful internal test storing kg_summary as derivative         |
| 4              | Optimization            | Performance tuning, caching, vector index optimization                                               | p95 latency & throughput targets met                              |
| 5 (Deferred)   | Derivatives GA          | /derivatives full CRUD (except hard delete), retention policies, access control tiers                | DR-001 & DR-002 closed; policy docs approved                      |


## 14. Open Questions
- Which workflow engine: bespoke vs Temporal vs Prefect?  
- Vector store selection: FAISS local vs managed (pgvector, Milvus).  
- Graph DB: Do we need property graph (Neo4j) or embedded representation enough?  
- Rate limits for external RAG system?  
- Human feedback storage model?  
 - (Deferred) Uploading evaluation derivatives back to KM (POST /documents/:id/derivatives) scope & retention policy (DR-001).  
 - (Deferred) Potential derivative resource types (chunk_index, kg_summary, evaluation_run_metrics, testset_schema) sharing core metadata {derivative_id, source_run_id, resource_type, created_at, hash, pii_classification}; final contract deferred (DR-002).  
     *Update:* Partial draft provided in Section 8.8; enums & retention still deferred.
 - (Tracking) KM DB integration initial scope limited to testset summary & KG summary exports (FR-041, FR-042); broader derivative push remains deferred.
 - (Resolved) KG visualization library selection → Cytoscape.js with lazy loading (see Design Section 20); impacts payload shape for GET /ui/kg/{kg_id}/summary.

## 15. Assumptions
- KM API provides consistent versioning & stable IDs.  
- RAG system accessible via internal network with predictable latency.  
- Container platform: Kubernetes (implied); otherwise Docker Compose for dev.  
- Language focus: Python (backend), TypeScript (portal).  

## 16. Risks & Mitigations
| Risk                              | Impact                              | Mitigation                            |
|-----------------------------------|-------------------------------------|---------------------------------------|
| LLM latency spikes                | Slower eval throughput              | Add concurrency controls + caching    |
| Metric drift due to model updates | Inconsistent historical comparisons | Version metrics + store model hash    |
| Over-fragmentation of services    | Operational overhead                | Define consolidation guardrail review |
| Ingestion duplication             | Storage bloat                       | Checksum + idempotency keys           |
| Plugin security risk              | Code execution vulnerability        | Sandbox / sign plugins                |

## 17. Recommendations (Enhancements Beyond Initial Thoughts)
- Adopt event bus (e.g., NATS / Kafka) for decoupled stage notifications.  
- Implement unified Job Status schema across services.  
- Introduce feature flags (OpenFeature) for gradual rollout of new metrics.  
- Provide SDK (Python) wrapping API calls for power users & backward-compatible CLI shim.  
- Add caching layer for repeated RAG queries during evaluation retries.  
- Support evaluation scenario replay using stored rag_contexts + prompts.  
- Implement deterministic testset seed handling for reproducibility.  

## 18. Traceability Matrix (Scaffold)
Note: KG visualization library open question is now resolved (Cytoscape.js, see Design Section 20); related UI-FR-016..018 remain gated by feature flag `kgVisualization`.
| Req ID     | Type    | Related Objective(s) | Origin Section     | Linked UI (if backend) | Verification Artifact / Test Ref            |
|------------|---------|----------------------|--------------------|------------------------|---------------------------------------------|
| FR-001     | Backend | Obj2, Obj3           | 9.1 Ingestion      | UI-FR-009,010          | T-Ingest-Create                             |
| FR-002     | Backend | Obj2                 | 9.1 Ingestion      | UI-FR-009              | T-Ingest-Checksum                           |
| FR-010     | Backend | Obj2                 | 9.3 KG             | UI-FR-016,017,018      | T-KG-Build-MinRelationships                 |
| FR-013     | Backend | Obj2, Obj4           | 9.4 Testset Gen    | UI-FR-019..022         | T-Testset-Generation-Config                 |
| FR-017     | Backend | Obj2, Obj4           | 9.5 Evaluation     | UI-FR-023..026         | T-Eval-Run-Lifecycle                        |
| FR-026     | Backend | Obj2                 | 9.7 Reporting      | UI-FR-030..032         | T-Report-HTML-Generation                    |
| FR-037     | Backend | Obj2                 | 9.10 Reporting Int | UI-FR-030..032         | run_meta.json contains report urls          |
| FR-041     | Backend | Obj4                 | 9.11 KM Storage    | UI-FR-033..035         | KM testset_summary export validation        |
| FR-042     | Backend | Obj4                 | 9.11 KM Storage    | UI-FR-033..035         | KM kg_summary export validation             |
| UI-FR-009  | UI      | Obj2                 | UI Req 5.3         | n/a                    | UI test: documents list renders checksum    |
| UI-FR-016  | UI      | Obj2, Obj4           | UI Req 5.5         | FR-010..012            | UI test: KG metrics panel                   |
| UI-FR-019  | UI      | Obj2, Obj4           | UI Req 5.6         | FR-013..016            | UI test: testset form hash display          |
| UI-FR-023  | UI      | Obj2, Obj4           | UI Req 5.7         | FR-017..022            | WebSocket stream integration test           |
| UI-FR-030  | UI      | Obj2                 | UI Req 5.9         | FR-026..027,037,040    | UI report preview test                      |
| UI-FR-033  | UI      | Obj4                 | UI Req 5.10        | FR-041,042             | UI KM summary delta highlight test          |
| UI-FR-060  | UI      | Obj4                 | UI Req 5.18        | FR-017..022            | Lineage chain display test                  |
| UI-FR-062  | UI      | Obj2 (migration)     | UI Req 5.19        | Multiple               | CLI hint rendering test                     |
| UI-FR-065  | UI      | Obj4 (future)        | UI Req 5.20        | DR-001/DR-002 deferred | Derivatives tab placeholder visible         |
| UI-NFR-001 | UI-NFR  | Obj3                 | UI NFR Section     | n/a                    | Availability synthetic check                |
| UI-NFR-005 | UI-NFR  | Obj4                 | UI NFR Section     | FR-031..033            | trace_id presence in 90% API calls (sample) |
| UI-NFR-006 | UI-NFR  | Obj2                 | UI NFR Section     | n/a                    | Bundle size CI budget                       |
| ...        | ...     | ...                  | ...                | ...                    | ...                                         |

## 19. Next Steps
1. Review & approve scope.  
2. Prioritize Phase 1 subset (FR 1-9, 17-22, 31-35, NFR core).  
3. Produce OpenAPI skeletons.  
4. Define event schemas & shared libraries.  
5. Establish CI templates for new services.  

---
End of document.
