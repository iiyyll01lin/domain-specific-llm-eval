# Implementation Plan – RAG Evaluation Platform & UI Lifecycle Console

Version: 0.1  
Status: Planning Draft  
Date: 2025-09-10  
Owner: Platform Engineering  

---
## 1. Purpose
Provide a structured, traceable implementation roadmap translating EARS requirements (`requirements.md`, `requirements.ui.md`) and design specifications (`design.md`, `design.ui.md`) into discrete, testable engineering tasks. Plan covers both backend microservices transformation and embedded UI Lifecycle Console integration inside `insights-portal/`.

## 2. Guiding Principles
- Traceability First: Every Functional Requirement (FR / UI-FR) mapped to ≥1 task.
- Incremental Hardening: Deliver baseline path end-to-end early (documents → report) before optimizations.
- Deterministic Reproducibility: Idempotency anchors implemented in earliest tasks for caching later.
- Observability By Default: Structured logs + metrics hooks from first runnable slice.
- Feature Flag Isolation: Optional features (KG visualization, multi-run compare) gated & lazy loaded.

## 3. Milestone Timeline (Indicative)
| Milestone                              | Sprint | Core Deliverables                                                                           |
|----------------------------------------|--------|---------------------------------------------------------------------------------------------|
| M1 Baseline Ingestion→Processing       | 1      | Ingestion svc, Processing svc, artifact persistence, basic UI tables (Documents/Processing) |
| M2 Testset & Evaluation Spine          | 2      | Testset Gen svc, Evaluation Runner svc, evaluation artifacts, UI testsets & runs panels     |
| M3 Reporting & KM Summaries            | 3      | Reporting svc, run_meta linking, KM export summaries + UI display                           |
| M4 Knowledge Graph (Optional Flag)     | 4      | KG builder svc, summary endpoint, UI KG summary + lazy visualization flag                   |
| M5 WebSocket & Performance Hardening   | 5      | WS multiplex (progress/update), polling scheduler, bundle budget guard                      |
| M6 Advanced Telemetry & Subgraph Draft | 6      | Telemetry taxonomy impl, subgraph API draft endpoint, manifest integrity prototype          |

## 3.1 Engineer Allocation & Timeline
| Engineer                            | Role Focus                                                                                    | Summary Scope                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| E1 (Backend & Infra)                | Core service scaffolding, ingestion, processing pipeline, storage, infra & security hardening | TASK-001~005, 010~016, 015a-d, support 033 I/O perf, later 090+ (not shown in this excerpt)       |
| E2 (Generation & Evaluation)        | Testset generation, evaluation runner, metrics registry, aggregation & reporting              | TASK-020a-d, 021a-d, 022~024, 030a-d, 031a-d, 032, 033a-d, 034 (not fully shown), 040~043 (later) |
| E3 (UI / Realtime / KG / Telemetry) | UI panels, polling → WS upgrade, KG optional feature, telemetry, bundle governance            | TASK-017, 025, 036, 044 (later), 060+ (KG later), 070+ (WS), 081/082 (later)                      |

### 3.1.1 Sprint-Level Distribution (Indicative)
| Sprint | Critical Path (Primary Owner)                                                  | Parallel Work (Owner)                                                                            | Notes                                                 |
|--------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| 1      | Ingestion & Processing backbone (E1: 001-005, 010-016, 015a-d)                 | UI Documents/Processing panel (E3: 017); Testset config hash & validation spike (E2: 020a, 020c) | Establish deterministic & observability anchors early |
| 2      | Testset full flow + Eval API (E2: 020b-d, 021a-d, 022-024, 030a-d)             | Processing hardening (E1: 015c/d robustness); UI Testsets & Runs panels (E3: 025, 036)           | End-to-end questions → evaluation items               |
| 3      | Metrics registry & aggregation + Reporting (E2: 031a-d, 032, 033a-d, 034, 035) | KM summary prep (E1 assist); Reports & Summaries UI (E3)                                         | Produces kpis.json & report artifacts                 |
| 4      | Optional KG service (E3 lead: 060-063, 062a-d) with extraction support (E1)    | Summary export integration (E2: 045)                                                             | Feature-flagged; avoid blocking earlier SLOs          |
| 5      | WebSocket gateway & downgrade logic (E3: 070-073)                              | Perf/backpressure tuning (E2: optimize 033/034); Expanded metrics (E1: 080)                      | Transition from polling to realtime                   |
| 6      | Telemetry taxonomy + subgraph draft (E3: future 066/067, 081, 118/119)         | Manifest integrity prototype (E2: 083)                                                           | Hardening & governance layer                          |

> NOTE: Downstream tasks beyond section 5.4 are summarized; only tasks present in this file segment are annotated below. Additional later-governance tasks should follow the same `engineer` and `target_sprint` pattern.

## 4. Task Identification Schema
| Field               | Meaning                                        |
|---------------------|------------------------------------------------|
| ID                  | TASK-### unique identifier                     |
| Title               | Concise action-oriented summary                |
| Description         | Implementation details & rationale             |
| Acceptance Criteria | Verifiable completion checks                   |
| Dependencies        | Blocking tasks / resources                     |
| Artifacts           | Files, endpoints, configs produced or modified |
| Req Mapping         | FR / UI-FR / NFR references                    |

### 4.1 Acceptance Criteria Template
Use structured, testable, and measurable patterns. Each criterion SHOULD map to at least one automated check (unit/integration/CI gate) when feasible.

| Type            | Template Pattern                                             | Example                                                       | Required Quantification                            |
|-----------------|--------------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------|
| Functional      | Given <precondition> When <action> Then <observable result>  | Given valid doc When POST /documents Then 202 and job_id UUID | Precise HTTP codes / schema fields                 |
| Idempotency     | Repeating <operation> N times yields same <artifact/hash/id> | Re-run config hash calc 3x -> identical hash string           | N ≥2 & equality condition                          |
| Determinism     | With seed=<value> first N outputs stable                     | Seed=42 first 5 questions identical                           | Define N + seed value                              |
| Performance     | p95 < THRESHOLD under baseline workload                      | Processing p95 <30s for 50k tokens                            | Include measurement method reference (workload.md) |
| Resilience      | Inject <fault> -> system <fallback/retry behavior>           | Simulated 429 -> 3 retries exponential backoff                | # retries, backoff caps                            |
| Observability   | Metric <name> emitted with labels <k=v>                      | processing_embedding_batch_duration_seconds present           | Metric name & at least one label                   |
| Security/Supply | Tool <scanner> exits 0 with no HIGH findings                 | Trivy scan no HIGH/CRITICAL                                   | Severity thresholds                                |
| Data Integrity  | <manifest/count/hash> matches persisted artifacts            | chunk_count == embedding_count                                | Exact field match                                  |
| Degradation     | After <N> failures, downgrade path activated                 | 2 missed heartbeats -> downgrade within 5s                    | N + reaction time                                  |

Checklist snippet (embed in DoD if needed):
```
- [ ] All acceptance statements follow a pattern (above table)
- [ ] At least one quantitative (p95 / hash / count) check
- [ ] Links to workload.md for any performance SLO
- [ ] Negative path / fault injection covered for resilience tasks
- [ ] Metric & log keys referenced exist (or added to taxonomy/design)
```

## 5. Detailed Task Catalog
### 5.1 Foundation & Repo Prep
| ID       | Title                             | Description                                                                                                                                 | Acceptance Criteria                                                     | Dependencies | Artifacts                      | Req Mapping           |
|----------|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------|--------------------------------|-----------------------|
| TASK-001 | Service Skeleton Scaffolding      | Create base FastAPI project structure for microservices (ingestion, processing, testset, eval, reporting, adapter) with shared lib package. | Repos build; uvicorn launch each svc; shared utilities imported.        | None         | services/*; services/pyproject.toml; services/tests/test_service_skeleton.py | FR baseline infra     |
| TASK-002 | Common Config & Env Loader        | Implement unified settings module (env + .env + defaults) and validation.                                                                   | All services log config at startup; missing critical var aborts.        | TASK-001     | services/common/config.py      | NFR robustness        |
| TASK-003 | Logging & Trace ID Middleware     | Structured JSON logs + per-request trace_id injection.                                                                                      | log line contains trace_id, path, status_code.                          | TASK-001     | middleware/logging.py          | NFR observability     |
| TASK-004 | Error Envelope Standardization    | Implement exception handlers returning {error_code,message,trace_id}.                                                                       | 4xx/5xx responses conform; tests validate.                              | TASK-003     | error_handlers.py              | UI-FR-053~055         |
| TASK-005 | Object Storage Client Abstraction | Wrapper for S3/MinIO ops with retry/backoff + checksum util.                                                                                | Upload/download integration tests pass; checksum mismatch raises error. | TASK-001     | storage/object_store.py        | FR ingest, processing |

```yaml
# TASK-001 Governance
governance:
	status: Completed
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-28
	verification:
		- pytest services/tests/test_service_skeleton.py -q
	deliverables:
		- services/pyproject.toml
		- services/tests/test_service_skeleton.py
		- services/common/config.py
		- services/*/main.py
	risk: "Inconsistent service layout hampers reuse"
	mitigation: "Template + scaffold test verifying directories"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	dod:
		- Health endpoints expose service-specific identifiers with trace_id header
		- Shared middleware and error handlers imported across services without runtime errors
		- Scaffold smoke test validates validation envelope contract

# TASK-002 Governance
governance:
	status: Completed
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-29
	risk: "Misconfigured env vars cause runtime instability"
	mitigation: "Central pydantic validation + fail-fast; config snapshot in logs"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests","lint-config"]
	verification:
		- pytest services/tests/test_config.py -q
		- pytest services/tests/test_service_skeleton.py -q
		- pytest tests/services/common/test_object_store.py -q
	dod:
		- Missing critical var abort test
		- Config printed once with redaction
		- .env.example updated

# TASK-003 Governance
governance:
	status: Completed
	engineer: E3   # Shared log format consumption by UI, but implemented by backend infra engineer if preferred
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-29
	risk: "Unstructured logs hinder incident triage"
	mitigation: "JSON logger + trace_id middleware test + structured schema"
	adr_impact: ["ADR-002"]
	ci_gate: ["unit-tests","log-schema-check"]
	verification:
		- pytest services/tests/test_logging.py services/tests/test_errors.py -q
	dod:
		- Log schema unit test
		- Trace id present in request log test
		- Error path logs include trace id

# TASK-004 Governance
governance:
	status: Completed
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-29
	risk: "Inconsistent error shapes break UI handling"
	mitigation: "Unified handler + contract test fixtures"
	adr_impact: ["ADR-003"]
	ci_gate: ["unit-tests","api-schema"]
	verification:
		- pytest services/tests/test_logging.py services/tests/test_errors.py -q
	dod:
		- 4xx/5xx error schema snapshot
		- Trace id always present
		- UI error parsing test passes

# TASK-005 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P2
	estimate: 2p
	risk: "Unreliable object storage leads to data loss"
	mitigation: "Retry + checksum mismatch test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/common/test_object_store.py
	dod:
		- Retry logic test
		- Checksum mismatch failure
		- README storage section
```

### 5.2 Ingestion & Processing
| ID       | Title                                         | Description                                                            | Acceptance Criteria                                                                                                      | Dependencies       | Artifacts                        | Req Mapping                  |
|----------|-----------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------|----------------------------------|------------------------------|
| TASK-010 | Ingestion API (POST /documents)               | Accept km_id + version, enqueue ingestion job.                         | 202 response; job row persisted.                                                                                         | TASK-001           | services/ingestion/main.py, services/ingestion/repository.py, services/ingestion/openapi.json | FR ingest                    |
| TASK-011 | KM Fetch + Checksum Pipeline                  | Stream download, calculate checksum, dedupe, store raw.                | Duplicate returns existing doc_id; raw stored once.                                                                      | TASK-010, TASK-005 | ingestion/worker.py              | FR ingest                    |
| TASK-012 | Ingestion Event Emission                      | Emit document.ingested (internal bus or simple pubsub).                | Event captured in test harness.                                                                                          | TASK-011           | events/schema.py                 | FR ingest, traceability      |
| TASK-013 | Processing Job API (POST /process-jobs)       | Launch processing referencing document_id + profile hash.              | 202 + job_id; validation on document existence.                                                                          | TASK-011           | processing/api.py                | FR processing                |
| TASK-014 | Text Extraction & Normalization Stage         | Implement extraction adapters (PDF/text) + unicode/whitespace cleanup. | Sample PDFs processed accurately (golden test).                                                                          | TASK-013           | processing/stages/extract.py     | FR processing                |
| TASK-015 | Chunking & Embeddings Stage                   | Token-based chunking + embedding generation batch logic.               | chunks.jsonl emitted; embedding count matches chunk count; single 50k token doc processing p95 <30s (baseline workload). | TASK-014           | processing/stages/chunk_embed.py | FR processing                |
| TASK-016 | Processing Completion Event                   | Emit document.processed with chunk_count + profile hash.               | Event contract validated.                                                                                                | TASK-015           | events/schema.py                 | FR processing, traceability  |
| TASK-017 | UI Integration: Documents & Processing Panels | Implement portal tables + polling scheduler (10s).                     | Displays documents + processing jobs with statuses.                                                                      | TASK-013, UI shell | insights-portal/src/...          | UI-FR-003/004, UI-FR-008~012 |

```yaml
# TASK-011 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion -q
	dod:
		- KM client streams document content and deduplicates by km/version and checksum
		- Object store upload occurs once per unique payload
		- Job status and error columns updated in repository schema
# TASK-012 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion/test_worker.py
	dod:
		- document.ingested envelope emitted for new documents
		- Duplicate ingestion requests reuse existing document without extra events
		- Event publisher injectable for unit tests
# TASK-013 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/processing/test_jobs_api.py
	dod:
		- POST /process-jobs returns 202 with persisted job metadata
		- Missing document_id responds with document_not_found envelope (404)
		- Invalid payload routes through standardized validation handler
# TASK-014 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-26
	verification:
		- pytest tests/services/processing/test_extraction_stage.py
	dod:
		- PDF and text adapters share normalization pipeline with whitespace collapse and Unicode cleanup
		- Unsupported binary payloads yield standardized unsupported_mime_type errors
		- Empty-text guard raises extraction_empty_text to protect downstream stages
# TASK-016 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 3p
	risk: "Event omission breaks downstream traceability"
	mitigation: "Worker orchestrator unit tests + schema validation"
	adr_impact: ["ADR-004","ADR-006"]
	ci_gate: ["unit-tests","event-contract"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_worker.py -q
	deliverables:
		- services/processing/worker.py
		- tests/services/processing/test_worker.py
		- services/common/events.py
		- events/schemas/document.processed.v1.json
	dod:
		- Worker orchestrates extract → chunk → embed → persist and updates job status transitions
		- document.processed event includes chunk_count, embedding_count, manifest_object_key, duration_ms
		- Schema fixtures synchronized across EN/ZH docs & README reference updated
# TASK-017 Governance
governance:
	status: Done
	engineer: E3
	target_sprint: 1
	owner: ui-platform@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-30
	verification:
		- vitest run src/app/lifecycle/__tests__/DocumentsPanel.test.tsx src/app/lifecycle/__tests__/ProcessingPanel.test.tsx
	dod:
		- Documents panel surfaces latest status, timestamps, and failure reasons with 10s polling via lifecycle config
		- Processing panel renders all jobs with status chips and manual refresh wired to polling hook
		- Polling hook updates lifecycle store while respecting AbortController cancellation and fetch timeouts
		- i18n resources supply zh-TW/en-US copy for lifecycle tabs and empty states
# TASK-015 Governance
governance:
	status: Done              # Planned | In-Progress | Blocked | Done | Verified
	engineer: E1
	target_sprint: 1
	owner: platform-ml@team
	priority: P0
	estimate: 5p
	risk: "Batch embedding concurrency may trigger GPU/RAM OOM or API rate limiting"
	mitigation: "Cap batch size 512; exponential backoff (3 retries); env MAX_EMB_BATCH override"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["build-governance:schemas","perf-baseline"]
	slo:
		embedding_stage_p95_seconds: 30
		embedding_error_rate_percent: 1.0
	metrics:
		- processing_embedding_batch_duration_seconds
		- processing_embedding_batch_size
		- processing_embedding_error_total
	logs:
		- code=EMBED_BATCH_START level=INFO
		- code=EMBED_BATCH_FAIL level=ERROR
	completed_at: 2025-09-30
	verification:
		- python3 -m pytest tests/processing -q
		- python3 -m pytest -q
	dod:
		- Unit tests cover tokenizer + batch executor failure paths
		- chunks.jsonl produced; chunk_count == embedding_count
		- Over-limit batch triggers downsize log
		- Metrics exposed at /metrics
		- README embedding section updated
		- Failure emits standardized error envelope
```

#### TASK-015 Subtasks
| Sub-ID    | Title                    | Description                                                                                | Acceptance Criteria                                                                   | Dependencies | Artifacts                           | Notes                   |
|-----------|--------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|--------------|-------------------------------------|-------------------------|
| TASK-015a | Tokenizer & Boundary     | Implement language-aware tokenizer + sentence/paragraph boundary detection with fallback.  | Mixed EN/ZH sample produces stable boundaries; fallback triggers on unsupported mime. | TASK-015     | processing/stages/tokenizer.py      | Provides tokens + spans |
| TASK-015b | Chunk Assembly Rules     | Assemble chunks by target token size (≈512) with configurable overlap (default 50 tokens). | No chunk > hard max (800); deterministic order for same seed/config.                  | TASK-015a    | processing/stages/chunk_rules.py    | Overlap & size config   |
| TASK-015c | Embedding Batch Executor | Batch executor with retry (exponential backoff), per-batch timeout, circuit breaker.       | 429/timeout retried ≤3; breaker opens after 5 consecutive failures & logs event.      | TASK-015b    | processing/stages/embed_executor.py | Emits metrics/logs      |
| TASK-015d | Persistence & Integrity  | Persist chunks.jsonl + embeddings; compute per-chunk SHA256 & counts manifest.             | Manifest counts match file; hash mismatch test fails fast.                            | TASK-015c    | processing/stages/chunk_persist.py  | Feeds downstream        |

```yaml
# TASK-015a Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	risk: "Tokenizer edge cases mis-split multilingual text"
	mitigation: "Golden mixed corpus tests + fallback path"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-26
	verification:
		- pytest tests/services/processing/test_tokenizer_stage.py
	dod:
		- Mixed English/Chinese corpus segmentation verified via unit tests
		- Unsupported mime path emits downgrade log and returns deterministic single segment
		- Sentence heuristics documented in services/processing/README.md
# TASK-015b Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 1p
	risk: "Oversized chunks degrade embedding latency"
	mitigation: "Hard cap + size histogram test"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_chunk_rules.py -q
	deliverables:
		- services/processing/stages/chunk_rules.py
		- tests/services/processing/test_chunk_rules.py
	dod:
		- No chunk >800
		- Determinism test
		- Size histogram emitted
		- Overlap trimmed when exceeding hard_max while preserving suffix order
# TASK-015c Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P0
	estimate: 3p
	risk: "Unbounded retries overload upstream"
	mitigation: "Max attempts + circuit breaker tests"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-30
	verification:
		- pytest tests/services/processing/test_embed_executor.py -q
	deliverables:
		- services/processing/stages/embed_executor.py
		- tests/services/processing/test_embed_executor.py
		- services/common/config.py
		- services/pyproject.toml
	dod:
		- Retry/backoff logic verified against timeout and 429 scenarios
		- Circuit breaker opens after consecutive failures and emits metric
		- Error envelope normalizes provider exceptions
		- Batch size capped via settings.processing_embedding_max_batch_size
		- Prometheus metrics exposed for durations, failures, and batch size
# TASK-015d Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	risk: "Integrity mismatch undetected"
	mitigation: "Count+hash manifest validation test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_chunk_persistence.py -q
	deliverables:
		- services/processing/stages/chunk_persist.py
		- tests/services/processing/test_chunk_persistence.py
	dod:
		- Manifest schema serialized alongside chunks.jsonl & embeddings.jsonl
		- Hash mismatch raises ChunkPersistenceError and aborts upload
		- README integrity notes describe manifest + checksum workflow
```

### 5.3 Testset Generation
| ID       | Title                                | Description                                                                         | Acceptance Criteria                                | Dependencies | Artifacts               | Req Mapping               |
|----------|--------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------|--------------|-------------------------|---------------------------|
| TASK-020 | Testset Job API (POST /testset-jobs) | Accept config; hash normalization for determinism.                                  | config_hash stable across field order; 202 job_id. | TASK-016     | testset/api.py          | FR-013~016                |
| TASK-021 | Question/Answer Synth Engine         | Implement core generation using existing RAGAS functions; seed for reproducibility. | Same seed + config yields same first N questions.  | TASK-020     | testset/engine.py       | FR-013~016                |
| TASK-022 | Persona & Scenario Generation        | Generate personas.json & scenarios.json with counts.                                | persona_count & scenario_count fields present.     | TASK-021     | testset/persona.py      | FR-013~016, UI-FR-019~022 |
| TASK-023 | Deduplication & Cap Enforcement      | MinHash or set-based dedupe + max sample limit prior to persist.                    | Duplicate ratio < threshold (config).              | TASK-022     | testset/dedupe.py       | FR-013~016                |
| TASK-024 | Testset Created Event                | Emit testset.created (id, sample_count).                                            | Event schema test passes.                          | TASK-023     | events/schema.py        | FR-013~016, traceability  |
| TASK-025 | UI Testsets Panel                    | Portal component with polling & config hash display.                                | Shows sample_count, seed, config_hash.             | TASK-024     | insights-portal/src/... | UI-FR-019~022             |
#### TASK-020 Subtasks
| Sub-ID    | Title                    | Description                                           | Acceptance Criteria                                                   | Dependencies | Artifacts                    | Notes                |
|-----------|--------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|--------------|------------------------------|----------------------|
| TASK-020a | Config Normalizer & Hash | Normalize config ordering & defaults then hash.       | Same logical config -> identical hash; empty optional fields ignored. | TASK-020     | testset/config_normalizer.py | Deterministic anchor |
| TASK-020b | Idempotent Job Guard     | Prevent duplicate active job for same config+version. | Second submit returns existing job_id within 50ms.                    | TASK-020a    | testset/job_guard.py         | Uses hash index      |
| TASK-020c | Validation Layer         | Schema + value range validation (counts, seed)        | Invalid field triggers 400 with error_code.                           | TASK-020a    | testset/validation.py        | Reusable             |
| TASK-020d | Audit Log & Metrics      | Emit creation metric + structured log (hash,size)     | Metric testset_job_created_total increments; log contains hash.       | TASK-020b    | testset/metrics.py           | Observability        |

```yaml
# TASK-021 Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 3p
	risk: "Seed instability breaks reproducibility"
	mitigation: "Generator + engine deterministic tests"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_engine.py -q
		- pytest services/tests/testset/test_generator_core.py -q
	deliverables:
		- services/testset/engine.py
		- services/testset/repository.py
		- services/tests/testset/test_engine.py
	dod:
		- Engine persists samples and metadata with deterministic object keys and checksums
		- Repository supports running/completed transitions with audit timestamps and error resets
		- Metadata document exposes persona/scenario counts, seed, and checksum for traceability
		- Unit tests cover success, missing job, and empty generation failure paths
# TASK-022 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
# TASK-023 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
# TASK-024 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
# TASK-025 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 2
# TASK-020a Governance
governance:
	status: Done
	engineer: E2
	target_sprint: 1
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "Hash drift from field order changes"
	mitigation: "Normalizer golden tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- python3 -m pytest services/tests/testset/test_config_normalizer.py
	deliverables:
		- services/testset/config_normalizer.py
		- services/tests/testset/test_config_normalizer.py
	dod:
		- Order invariance tests
		- Empty optional stripped
		- Hash doc added
# TASK-020b Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py::test_duplicate_submission_returns_same_job_id -q
	dod:
		- Guard ensures duplicate config_hash returns existing job within same request cycle
		- Repository enforces UNIQUE index on config_hash for persistence-level safety
		- Duplicate guard path emits structured debug log with job_id and status
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "Duplicate jobs waste compute"
	mitigation: "Race condition test with threads"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Duplicate returns existing id
		- Concurrency test pass
		- Structured dedupe log
# TASK-020c Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 1
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-30
	risk: "Invalid counts produce runaway generation"
	mitigation: "Range validation + negative tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- python3 -m pytest services/tests/testset/test_validation.py -q
	deliverables:
		- services/testset/validation.py
		- services/tests/testset/test_validation.py
	dod:
		- Config validation prevents submission without required fields
		- Invalid counts rejected with 400 and error_code testset_config_invalid
		- Seed and sample limits enforced with unit coverage
		- Selected strategies de-duplicated and persona metadata normalised
# TASK-020d Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "Missing audit trail reduces traceability"
	mitigation: "Metric + log integration tests"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py -q
	dod:
		- Prometheus counter labels capture created vs duplicate outcomes
		- Structured log includes job_id, config_hash, method for observability
		- /metrics endpoint exposes testset_job_created_total for scrape
```

#### TASK-021 Subtasks
| Sub-ID    | Title                          | Description                                           | Acceptance Criteria                              | Dependencies | Artifacts                     | Notes             |
|-----------|--------------------------------|-------------------------------------------------------|--------------------------------------------------|--------------|-------------------------------|-------------------|
| TASK-021a | Seeded Generator Core          | Core Q/A generation loop with seed control.           | Fixed seed yields identical first 5 Q/A pairs.   | TASK-021     | testset/generator_core.py     | Deterministic     |
| TASK-021b | Persona Injection Layer        | Apply persona context modifiers to prompts.           | persona tokens appended; snapshot diff approved. | TASK-021a    | testset/persona_injector.py   | Extensible        |
| TASK-021c | Scenario Variation Module      | Alter context per scenario rules.                     | At least 3 scenario variants per doc group.      | TASK-021b    | testset/scenario_variation.py | Diversity         |
| TASK-021d | Quality & Duplicate Pre-Filter | Light fuzzy match + length bounds before full dedupe. | Removes ≥90% trivial duplicates in test corpus.  | TASK-021c    | testset/pre_filter.py         | Performance guard |

```yaml
# TASK-021a Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-30
	risk: "Seed nondeterminism breaks reproducibility"
	mitigation: "Fixed seed fixture tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/testset/test_generator_core.py -q
	deliverables:
		- services/testset/generator_core.py
		- services/testset/payloads.py
		- services/tests/testset/test_generator_core.py
	dod:
		- First 5 Q/A stable
		- Seed param documented
		- Snapshot approved
# TASK-021b Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "Persona injection bloats prompts"
	mitigation: "Token length threshold test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_persona_injector.py -q
	deliverables:
		- services/testset/persona_injector.py
		- services/tests/testset/test_persona_injector.py
	dod:
		- Token overhead < limit
		- Snapshot diff updated
		- README persona section
# TASK-021c Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "Insufficient scenario diversity"
	mitigation: "Variant count assertion"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_scenario_variation.py -q
	deliverables:
		- services/testset/scenario_variation.py
		- services/tests/testset/test_scenario_variation.py
	dod:
		- ≥3 variants test
		- Diversity doc
		- Scenario rules README
# TASK-021d Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "Low quality or duplicates pass filter"
	mitigation: "Fuzzy similarity threshold tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_pre_filter.py -q
	deliverables:
		- services/testset/pre_filter.py
		- services/tests/testset/test_pre_filter.py
	dod:
		- Duplicate reduction >=90%
		- Length bounds enforced
		- Pre-filter metrics
```

```yaml
# TASK-010 Governance
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-ingestion@team
	priority: P1
	estimate: 2p
	risk: "Missing validation allows bad documents"
	mitigation: "Pydantic schema + negative tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion/test_documents_api.py
	dod:
		- 202 + job_id test
		- Invalid payload 400
		- OpenAPI doc updated

# TASK-020 Governance
governance:
	status: Completed
	engineer: E2
	target_sprint: 1
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "Config hash instability breaks caching"
	mitigation: "Normalizer golden test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py -q
		- pytest services/tests/testset/test_validation.py -q
	dod:
		- POST /testset-jobs returns 202 with job_id, config_hash, method metadata
		- Duplicate submissions reuse existing job_id and set duplicate=true
		- Metrics counter labels (created/duplicate) increment per request outcome
		- Structured log emits job_id and config_hash for traceability
```

### 5.4 Evaluation Runner
| ID       | Title                            | Description                                                                             | Acceptance Criteria                              | Dependencies | Artifacts                | Req Mapping              |
|----------|----------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------|--------------|--------------------------|--------------------------|
| TASK-030 | Evaluation Run API               | Create run referencing testset_id + rag profile.                                        | 202 run_id; validation on testset existence.     | TASK-024     | eval/api.py              | FR-017~022               |
| TASK-031 | RAG Invocation & Context Capture | Adapter layer to call target RAG system; collect contexts.                              | All evaluation_items contain context array.      | TASK-030     | eval/rag_adapter.py      | FR-017~022               |
| TASK-032 | Metrics Plugin Registry          | Load baseline metrics (faithfulness, answer_relevancy, precision etc.) via entrypoints. | registry lists metrics; each executed per item.  | TASK-031     | eval/metrics/__init__.py | FR-017~022               |
| TASK-033 | Evaluation Items Persistence     | Stream write evaluation_items.json with flush intervals.                                | File incremental growth; final count == samples. | TASK-032     | eval/persist.py          | FR-017~022               |
| TASK-034 | KPI Aggregation & kpis.json      | Aggregate metrics distribution & store.                                                 | p95/p50 values correct on test data.             | TASK-033     | eval/aggregate.py        | FR-017~022               |
| TASK-035 | run.completed Event              | Emit run.completed with counts & metrics_version.                                       | Event schema validated.                          | TASK-034     | events/schema.py         | FR-017~022, traceability |
| TASK-036 | UI Evaluation Runs Panel         | Display progress, verdict (if present), error_count.                                    | Poll updates reflect run state.                  | TASK-035     | insights-portal/src/...  | UI-FR-023~026            |
#### TASK-030 Subtasks
| Sub-ID    | Title                     | Description                                             | Acceptance Criteria                                     | Dependencies | Artifacts          | Notes         |
|-----------|---------------------------|---------------------------------------------------------|---------------------------------------------------------|--------------|--------------------|---------------|
| TASK-030a | Run Model & States        | Define run state enum + lifecycle transitions.          | Illegal transition rejected; state diagram test passes. | TASK-030     | eval/run_states.py | Governance    |
| TASK-030b | Input Validation Layer    | Validate testset_id & profile existence.                | Invalid profile returns 400 with error_code.            | TASK-030a    | eval/validation.py | Shared reuse  |
| TASK-030c | Idempotent Run Submission | Prevent duplicate active run with same testset+profile. | Second identical request returns same run_id.           | TASK-030b    | eval/run_guard.py  | Determinism   |
| TASK-030d | Metrics & Logging         | Emit run_created metric + structured log.               | run_created_total increments; log has run_id & profile. | TASK-030c    | eval/metrics.py    | Observability |

```yaml
# TASK-031 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
# TASK-033 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
# TASK-034 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
# TASK-035 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
# TASK-036 Governance
governance:
	status: Completed
	engineer: E3
	target_sprint: 3
	completed_at: 2025-09-30
# TASK-030a Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	verification:
		- pytest services/tests/testset -q
	deliverables:
		- services/testset/generator_core.py
		- services/testset/persona_injector.py
		- services/testset/scenario_variation.py
		- services/testset/pre_filter.py
		- services/testset/payloads.py
		- services/tests/testset/
	estimate: 1p
	risk: "State explosion or invalid transition logic"
	mitigation: "Finite enum + transition table test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Transition matrix unit tested
		- Invalid transition raises error
		- README updated with lifecycle
# TASK-030b Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Missing validation permits bad run config"
	mitigation: "Pydantic schema + negative tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Validation rejects bad profile
		- Tests cover missing testset_id
		- Error envelope documented
# TASK-030c Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Duplicate runs waste resources"
	mitigation: "Hash guard + race test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Duplicate request returns same run_id
		- Concurrency test passes
		- Log includes dedupe=true flag
# TASK-030d Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Missing metrics hinder observability"
	mitigation: "Metric name lint + sample scrape test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","build-governance:schemas"]
	dod:
		- Metric exposed
		- Structured log asserts
		- Docs updated
```

#### TASK-031 Subtasks
| Sub-ID    | Title                   | Description                                 | Acceptance Criteria                                   | Dependencies | Artifacts               | Notes              |
|-----------|-------------------------|---------------------------------------------|-------------------------------------------------------|--------------|-------------------------|--------------------|
| TASK-031a | Adapter Interface       | Define interface for RAG system invocation. | Missing method raises NotImplemented in tests.        | TASK-031     | eval/rag_interface.py   | Contract           |
| TASK-031b | Context Capture Wrapper | Wrap calls to store retrieved contexts.     | Each evaluation item includes context array length>0. | TASK-031a    | eval/context_capture.py | Adds observability |
| TASK-031c | Retry & Timeout Policy  | Implement retry w/ jitter + timeout.        | 429/timeouts retried ≤3; abort logs final error.      | TASK-031b    | eval/retry_policy.py    | Resilience         |
| TASK-031d | Metrics & Trace IDs     | Emit latency histogram & attach trace ids.  | rag_request_latency_seconds histogram present.        | TASK-031c    | eval/metrics.py         | Perf visibility    |

```yaml
# TASK-031a Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Interface drift breaks adapters"
	mitigation: "Abstract base + contract test"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	dod:
		- Missing method test
		- Contract docstring
		- Example adapter stub
# TASK-031b Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Context not captured reduces evaluation quality"
	mitigation: "Wrapper test ensures contexts length>0"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Context array persisted
		- Empty retrieval test
		- Docs updated
# TASK-031c Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P0
	estimate: 2p
	risk: "Unbounded retries cause latency spikes"
	mitigation: "Max attempts + jitter backoff test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- Retry policy tested
		- Timeout enforced
		- Metrics recorded
# TASK-031d Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Latency visibility gap"
	mitigation: "Histogram + scrape test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Histogram present
		- Trace id in log
		- README metrics updated
```

#### TASK-033 Subtasks
| Sub-ID    | Title                 | Description                                             | Acceptance Criteria                              | Dependencies | Artifacts             | Notes         |
|-----------|-----------------------|---------------------------------------------------------|--------------------------------------------------|--------------|-----------------------|---------------|
| TASK-033a | Stream Writer Core    | Incremental append with flush interval.                 | File grows; flush interval ≤ configured seconds. | TASK-033     | eval/stream_writer.py | Efficiency    |
| TASK-033b | Backpressure Handling | Apply queue with max size & drop policy.                | Simulated slow disk triggers backpressure log.   | TASK-033a    | eval/backpressure.py  | Stability     |
| TASK-033c | Integrity Manifest    | Maintain count & checksum manifest.                     | Final counts match; mismatch test fails build.   | TASK-033b    | eval/manifest.py      | Traceability  |
| TASK-033d | Metrics Emission      | Expose items_written counter + flush latency histogram. | Metrics visible at /metrics endpoint.            | TASK-033c    | eval/metrics.py       | Observability |

```yaml
# TASK-033a Governance
governance:
	status: Planned
	engineer: E2 (E1 support for I/O tuning)
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "I/O flush delays cause data loss on crash"
	mitigation: "Flush interval test + fsync option"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Flush interval honored
		- Growth test
		- README streaming notes
# TASK-033b Governance
governance:
	status: Planned
	engineer: E2 (E1 support)
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Backpressure not handled → OOM"
	mitigation: "Bounded queue + drop metric"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Slow disk simulation test
		- Drop policy documented
		- Metric asserted
# TASK-033c Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Mismatch counts unnoticed"
	mitigation: "Manifest validation test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Manifest schema test
		- Mismatch failure test
		- Doc section added
# TASK-033d Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Missing metrics reduce ops insight"
	mitigation: "Metric name lint"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Counters/hist present
		- Scrape test
		- README update
```

#### TASK-034 Subtasks
| Sub-ID    | Title                   | Description                                               | Acceptance Criteria                                       | Dependencies | Artifacts                 | Notes         |
|-----------|-------------------------|-----------------------------------------------------------|-----------------------------------------------------------|--------------|---------------------------|---------------|
| TASK-034a | Distribution Calculator | Compute p50, p95, min, max.                               | Test fixture results match expected values.               | TASK-034     | eval/distribution.py      | Core logic    |
| TASK-034b | Aggregation Integrity   | Validate metric schema & missing values fill (NaN guard). | NaN replaced with null; schema validation passes.         | TASK-034a    | eval/aggregation_guard.py | Data hygiene  |
| TASK-034c | KPI File Writer         | Write kpis.json atomically.                               | Temp file rename; partial write test prevented.           | TASK-034b    | eval/kpi_writer.py        | Reliability   |
| TASK-034d | Metrics Publication     | Publish aggregation duration + counter.                   | aggregation_duration_seconds & kpi_records_total present. | TASK-034c    | eval/metrics.py           | Observability |

```yaml
# TASK-034a Governance
governance:
	status: Planned
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "Incorrect percentile calc"
	mitigation: "Deterministic fixture tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Percentile tests
		- Negative values handled
		- Doc update
	engineer: E2
	target_sprint: 3
# TASK-034b Governance
governance:
	status: Planned
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "NaN values propagate to UI"
	mitigation: "Guard replaces NaN with null"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- NaN guard test
		- Schema validation
		- README hygiene note
# Add engineer assignment and sprint
	engineer: E2
	target_sprint: 3
# TASK-034c Governance
governance:
	status: Planned
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Partial write corrupts KPI file"
	mitigation: "Atomic rename test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Atomic write test
		- Temp file cleanup
		- Doc updated
# Add engineer assignment and sprint
	engineer: E2
	target_sprint: 3
# TASK-034d Governance
governance:
	status: Planned
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "Aggregation latency opaque"
	mitigation: "Duration metric + test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Duration metric present
		- Records counter present
		- README metrics updated
	engineer: E2
	target_sprint: 3
```

```yaml
# TASK-030 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	estimate: 2p
	risk: "Run lifecycle ambiguity"
	mitigation: "State diagram + transition tests"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	dod:
		- State enum defined
		- Invalid transition test
		- Lifecycle doc
```

```yaml
# TASK-032 Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P1
	estimate: 3p
	risk: "Plugin init exception could halt entire evaluation run"
	mitigation: "Per-metric try/except isolation; degraded mode logs error & continues"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["unit-tests","lint","build-governance:schemas"]
	plugin_contract_version: 1
	failure_isolation: try-except per metric with noop fallback
	slo:
		registry_init_seconds_p95: 1.5
		plugin_failure_rate_percent: 0.5
	metrics:
		- eval_metrics_registry_load_seconds
		- eval_metric_execution_duration_seconds
		- eval_metric_failure_total
	logs:
		- code=METRIC_PLUGIN_REGISTERED level=INFO
		- code=METRIC_PLUGIN_FAILED level=ERROR
	dod:
		- ≥3 baseline metrics (faithfulness, answer_relevancy, precision) load
		- Single faulty plugin does not block others
		- Deterministic alphabetical registry ordering
		- /metrics exposes load duration + failure counters
		- README metrics section documents plugin interface
	engineer: E2
	target_sprint: 3
```

#### TASK-032 Subtasks
| Sub-ID    | Title                         | Description                                                                            | Acceptance Criteria                                                            | Dependencies | Artifacts                  | Notes            |
|-----------|-------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------|----------------------------|------------------|
| TASK-032a | Loader Contract               | Define MetricPlugin interface + version constant; enforce required methods.            | Missing method raises clear error; version exposed via registry endpoint.      | TASK-032     | eval/metrics/interface.py  | Contract pinned  |
| TASK-032b | Baseline Metrics Impl         | Implement faithfulness, answer_relevancy, precision metrics with shared helpers.       | All metrics return numeric score in test harness; deterministic on fixed seed. | TASK-032a    | eval/metrics/baseline/*.py | Core metrics     |
| TASK-032c | Discovery & Failure Isolation | Implement plugin discovery (filesystem/entrypoints) + per-plugin try/except isolation. | Faulty sample plugin skipped; failure counter increments; others execute.      | TASK-032b    | eval/metrics/loader.py     | Isolation proven |

```yaml
# TASK-032a Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P1
	estimate: 1p
	risk: "Interface ambiguity causes plugin failures"
	mitigation: "Contract test + docstring"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	dod:
		- Interface test
		- Missing method error
		- Contract docs
	engineer: E2
	target_sprint: 2
# TASK-032b Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P1
	estimate: 2p
	risk: "Baseline metrics produce inconsistent scores"
	mitigation: "Deterministic fixture + seed control"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- 3 metrics implemented
		- Determinism test
		- README metrics list
	engineer: E2
	target_sprint: 3
# TASK-032c Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P1
	estimate: 2p
	risk: "Faulty plugin crashes registry"
	mitigation: "Isolation try/except tests"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Fault injection test
		- Skip warning log
		- Failure metric increments
	engineer: E2
	target_sprint: 3
```

### 5.5 Insights Adapter & Reporting
| ID       | Title                          | Description                                                | Acceptance Criteria                                      | Dependencies       | Artifacts                  | Req Mapping               |
|----------|--------------------------------|------------------------------------------------------------|----------------------------------------------------------|--------------------|----------------------------|---------------------------|
| TASK-040 | Insights Normalization Module  | Translate evaluation artifacts to portal-friendly schema.  | export_summary.json produced behind flag.                | TASK-035           | adapter/normalize.py       | FR-038/039                |
| TASK-041 | Reporting HTML Templates       | Executive + technical templates with metrics placeholders. | HTML renders deterministic snapshot.                     | TASK-040           | reporting/templates/*.html | FR-037~040                |
| TASK-042 | PDF Generation (Playwright)    | Headless convert HTML to PDF, update run_meta.json.        | pdf_url present; file valid size>0.                      | TASK-041           | reporting/pdf.py           | FR-037~040                |
| TASK-043 | report.completed Event         | Emit after successful PDF + HTML generation.               | Event validated in tests.                                | TASK-042           | events/schema.py           | FR-037~040                |
| TASK-044 | UI Reports Panel               | List reports with html/pdf availability + fallback logic.  | 404 PDF fallback to HTML visible.                        | TASK-043           | insights-portal/src/...    | UI-FR-030~032             |
| TASK-045 | KM Summary Export (testset/kg) | Produce minimal summaries (counts only).                   | testset_summary_v0 & kg_summary_v0 JSON schema validate. | TASK-024, TASK-060 | adapter/km_export.py       | FR-041/042, UI-FR-033~035 |
| TASK-046 | UI KM Summaries Panel          | Display summaries + delta (prev run).                      | Delta computation correct for added counts.              | TASK-045           | insights-portal/src/...    | UI-FR-033~035             |

### 5.6 Knowledge Graph (Flagged Feature)
```yaml
# TASK-040 Governance
governance:
	status: Planned
	owner: platform-reporting@team
	priority: P1
	estimate: 2p
	risk: "Schema drift between pipeline & UI"
	mitigation: "Golden sample snapshot test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- export_summary.json snapshot
		- Flag off by default
		- README usage
	engineer: E2
	target_sprint: 3
# TASK-041 Governance
governance:
	status: Planned
	owner: platform-reporting@team
	priority: P1
	estimate: 2p
	risk: "Template variable mismatch"
	mitigation: "Template render test matrix"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Executive template passes test
		- Technical template passes test
		- Placeholder coverage 100%
	engineer: E2
	target_sprint: 3
# TASK-042 Governance
governance:
	status: Planned
	owner: platform-reporting@team
	priority: P1
	estimate: 1p
	risk: "PDF render flakiness"
	mitigation: "Deterministic viewport & font embedding"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- PDF size >0 test
		- run_meta updated
		- Fallback logic documented
	engineer: E2
	target_sprint: 3
# TASK-043 Governance
governance:
	status: Planned
	owner: platform-reporting@team
	priority: P2
	estimate: 1p
	risk: "Event emitted before persistence"
	mitigation: "Order test ensures after write"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Event schema test
		- Emission ordering test
		- README event docs
	engineer: E2
	target_sprint: 3
# TASK-044 Governance
governance:
	status: Planned
	owner: platform-ui@team
	priority: P2
	estimate: 2p
	risk: "UI fails gracefully without PDF"
	mitigation: "Fallback integration test"
	adr_impact: []
	ci_gate: ["ui-tests"]
	dod:
		- Fallback screenshot test
		- Loading skeleton test
		- Accessibility scan
	engineer: E3
	target_sprint: 4
# TASK-045 Governance
governance:
	status: Planned
	owner: platform-reporting@team
	priority: P2
	estimate: 1p
	risk: "Summary counts inaccurate"
	mitigation: "Cross-check vs raw artifacts test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- testset_summary_v0 schema test
		- kg_summary_v0 schema test
		- README summary format
	engineer: E2
	target_sprint: 4
```
| ID       | Title                                    | Description                                                      | Acceptance Criteria                                     | Dependencies | Artifacts                                        | Req Mapping           |
|----------|------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------|--------------|--------------------------------------------------|-----------------------|
| TASK-060 | KG Builder Service API                   | Accept build request, schedule entity & relationship extraction. | 202 kg_id; status endpoint returns progress.            | TASK-015     | kg/api.py                                        | UI-FR-016~018         |
| TASK-061 | Entity & Keyphrase Extraction            | spaCy + KeyBERT hybrid; fallback segmentation.                   | Entities present on sampled nodes test set.             | TASK-060     | kg/extract.py                                    | UI-FR-016~018         |
| TASK-062 | Relationship Builders Integration        | Jaccard, Overlap, Cosine, SummaryCosine thresholds.              | relationships array non-empty on sample CSV docs.       | TASK-061     | kg/relationships.py                              | UI-FR-016~018         |
| TASK-063 | KG Summary Endpoint                      | Provide counts, degree histogram, top entities (limit).          | JSON schema passes; histogram bins <=50.                | TASK-062     | kg/summary.py                                    | UI-FR-016~018         |
| TASK-064 | UI KG Summary Panel                      | Show counts, histogram, top entities when flag true.             | Fallback text when flag false.                          | TASK-063     | insights-portal/src/...                          | UI-FR-016~018         |
| TASK-065 | Lazy Visualization Component (Cytoscape) | Dynamic import graph view; sampling cap 500 nodes.               | Bundle diff shows isolated chunk; renders test graph.   | TASK-064     | insights-portal/src/components/KgVisualization/* | UI-FR-018, UI-NFR-006 |
| TASK-066 | Subgraph API Draft Implementation        | Implement read-only deterministic sampled subgraph endpoint.     | Returns truncated flag & stable subset across calls.    | TASK-063     | kg/subgraph.py                                   | Spec §27              |
| TASK-067 | UI Subgraph Focus Interaction            | Add entity focus form to request subgraph & overlay.             | Pill shows sampling/truncated; errors surfaced cleanly. | TASK-066     | insights-portal/src/...                          | Future enhancement    |

#### TASK-062 Subtasks
```yaml
# TASK-046 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-060 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-061 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-062 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-063 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-064 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-066 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 4
# TASK-067 Governance
governance:
	status: Planned
	owner: platform-ui@team
	priority: P3
	estimate: 2p
	risk: "Sampling UX confusion"
	mitigation: "Truncated pill + docs"
	adr_impact: []
	ci_gate: ["ui-tests"]
	dod:
		- Focus form test
		- Overlay renders test
		- Accessibility check
	engineer: E3
	target_sprint: 6
```
| Sub-ID    | Title                       | Description                                                                 | Acceptance Criteria                                                      | Dependencies | Artifacts                    | Notes                  |
|-----------|-----------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|--------------|------------------------------|------------------------|
| TASK-062a | Node Property Enrichment    | Populate entities, keyphrases, summary, embeddings (if available) on nodes. | Sample KG nodes contain required properties; missing embeddings flagged. | TASK-062     | kg/extract.py                | Prereq for similarity  |
| TASK-062b | Jaccard & Overlap Layers    | Implement Jaccard (entities) + Overlap (keyphrases) relationship builders.  | Relationships >0 on sample CSV set; thresholds configurable.             | TASK-062a    | kg/relationships.py          | Non-embedding paths    |
| TASK-062c | Embedding Cosine & Fallback | Compute cosine similarity if embeddings exist else skip gracefully.         | No failure when embedding absent; similarity list length logged.         | TASK-062b    | kg/relationships.py          | Optional quality boost |
| TASK-062d | Threshold Tuning Harness    | Script to evaluate relationship counts vs threshold grid & output metrics.  | Harness produces JSON report with counts & avg similarity.               | TASK-062c    | scripts/kg_threshold_tune.py | Guides tuning          |

```yaml
# TASK-062a Governance
governance:
	status: Planned
	owner: platform-kg@team
	priority: P1
	estimate: 2p
	risk: "Missing node properties reduces relationship quality"
	mitigation: "Property completeness test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Sample node includes entities/keyphrases
		- Missing embedding flag
		- README KG props
	engineer: E3 (E1 assist extraction perf)
	target_sprint: 4
# TASK-062b Governance
governance:
	status: Planned
	owner: platform-kg@team
	priority: P1
	estimate: 1p
	risk: "Thresholds too strict yield zero links"
	mitigation: "Param sweep test harness"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- >0 relationships sample
		- Threshold config doc
		- Log includes counts
	engineer: E3
	target_sprint: 4
# TASK-062c Governance
governance:
	status: Planned
	owner: platform-kg@team
	priority: P2
	estimate: 1p
	risk: "No-embedding path throws errors"
	mitigation: "Skip logic test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Graceful skip test
		- Count of skipped nodes
		- README fallback
	engineer: E3
	target_sprint: 4
# TASK-062d Governance
governance:
	status: Planned
	owner: platform-kg@team
	priority: P2
	estimate: 1p
	risk: "Poor threshold guidance"
	mitigation: "JSON report aggregation test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Report includes counts & averages
		- Example tuning doc
		- Script usage README
	engineer: E3
	target_sprint: 4
```

```yaml
# TASK-065 Governance
governance:
	status: Planned
	owner: platform-kg@team
	priority: P2
	estimate: 2p
	risk: "Graph render bundle bloat"
	mitigation: "Lazy chunk + size diff test"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- Dynamic import works
		- Bundle size diff recorded
		- README visualization section
	engineer: E3
	target_sprint: 4
```

### 5.7 WebSocket & Realtime
| ID       | Title                           | Description                                               | Acceptance Criteria                                                                                                                                                     | Dependencies       | Artifacts                                   | Req Mapping        |
|----------|---------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|---------------------------------------------|--------------------|
| TASK-070 | WS Gateway Endpoint (Multiplex) | Single /ui/events websocket implementing handshake.       | Handshake success; unsupported topic rejected; avg reconnect time <2s (baseline workload); heartbeat every 15s tolerates 2 consecutive misses before downgrade trigger. | TASK-036, TASK-044 | ws/gateway.py                               | UI-FR-049~051      |
| TASK-071 | Event Envelope & Sequencing     | Implement seq, heartbeat, gap detection logic.            | Gap triggers REST resync in tests.                                                                                                                                      | TASK-070           | ws/envelope.py                              | UI-FR-049~051, §23 |
| TASK-072 | Client Hook useEventStream      | React hook managing connect/reconnect/backoff & dispatch. | Simulated disconnect recovers & resumes events.                                                                                                                         | TASK-071           | insights-portal/src/hooks/useEventStream.ts | UI-FR-049~051      |
| TASK-073 | Progressive Downgrade Logic     | After N failures revert to polling for 2 mins.            | E2E test shows downgrade + later upgrade.                                                                                                                               | TASK-072           | insights-portal/src/hooks/*                 | UI-FR-049~051      |

```yaml
# TASK-070 Governance
governance:
	status: Planned
	owner: platform-realtime@team
	priority: P1
	estimate: 2p
	risk: "Unbounded reconnect storms"
	mitigation: "Backoff + heartbeat downgrade test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- Handshake test
		- Heartbeat miss downgrade
		- Reconnect metric present
	engineer: E3
	target_sprint: 5
```
```yaml
# TASK-071 Governance
governance:
	status: Planned
	owner: platform-realtime@team
	priority: P1
	estimate: 2p
	risk: "Sequence gap logic race"
	mitigation: "Deterministic gap fixture"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Gap detection test
		- Heartbeat interval test
		- Resync fallback test
	engineer: E3
	target_sprint: 5
```

```yaml
# TASK-072 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 5
# TASK-073 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 5
```

### 5.8 Telemetry & Observability
| ID       | Title                              | Description                                               | Acceptance Criteria                                      | Dependencies | Artifacts                       | Req Mapping                          |
|----------|------------------------------------|-----------------------------------------------------------|----------------------------------------------------------|--------------|---------------------------------|--------------------------------------|
| TASK-080 | Metrics Instrumentation (Services) | Expose Prometheus endpoints; basic counters & histograms. | /metrics returns ingestion, processing counters.         | TASK-010     | services/*/metrics.py           | NFR observability                    |
| TASK-081 | Client Telemetry Logger            | logEvent helper & batching for high-frequency events.     | ui.kg.render & ui.ws.connect events recorded.            | TASK-072     | insights-portal/src/telemetry/* | §25 taxonomy                         |
| TASK-082 | Bundle Size & Perf Guard CI        | CI script enforcing chunk size budgets & diff thresholds. | Build fails if KG chunk >300KB gz.                       | TASK-065     | .github/workflows/ci.yml        | UI-NFR-006                           |
| TASK-083 | Manifest Integrity Prototype       | Generate manifest.json w/ sha256 per artifact.            | manifest lists all run artifacts & passes checksum test. | TASK-034     | eval/manifest.py                | Future hook (Section 28 suggestions) |
| TASK-084 | Event Schema JSON Validation       | JSON Schema for WS envelope + runtime validation toggle.  | Invalid event dropped & counter increments.              | TASK-071     | ws/schema.py                    | Reliability                          |

```yaml
# TASK-080 Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P1
	estimate: 1p
	risk: "Lack of base metrics hides regressions"
	mitigation: "Metrics presence test"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- /metrics exposes counters
		- Histogram test
		- README metrics listing
	engineer: E1 (E2 review metric names)
	target_sprint: 5

# TASK-082 Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P2
	estimate: 1p
	risk: "Bundle size creep unnoticed"
	mitigation: "CI budget script + diff test"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- Budget fail test
		- Diff report artifact
		- Docs budget policy
	engineer: E3
	target_sprint: 5

# TASK-083 Governance
governance:
	status: Planned
	owner: platform-observability@team
	priority: P2
	estimate: 1p
	risk: "Artifact drift undetected"
	mitigation: "Manifest schema + checksum tests"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Manifest schema test
		- Missing artifact failure
		- README manifest section
	engineer: E2
	target_sprint: 6
```
```yaml
# TASK-081 Governance
governance:
	status: Planned
	owner: platform-ui@team
	priority: P2
	estimate: 2p
	risk: "High-frequency events overwhelm backend"
	mitigation: "Batch size & flush interval tests"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Batching test
		- Event type coverage list
		- README telemetry usage
	engineer: E3
	target_sprint: 6
```
```yaml
# TASK-084 Governance
governance:
	status: Planned
	owner: platform-realtime@team
	priority: P1
	estimate: 1p
	risk: "Invalid events pollute state"
	mitigation: "Schema validation + counter"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Invalid drop test
		- Counter increment test
		- Config toggle documented
	engineer: E3
	target_sprint: 5
```

### 5.9 Security & Privacy (Phase 2 Prep)
| ID       | Title                           | Description                                             | Acceptance Criteria                                 | Dependencies | Artifacts                    | Req Mapping   |
|----------|---------------------------------|---------------------------------------------------------|-----------------------------------------------------|--------------|------------------------------|---------------|
| TASK-090 | Auth Stub & Token Injection     | Pluggable auth layer; no-op in dev.                     | Setting token env var injects Authorization header. | TASK-001     | services/common/auth.py      | Future NFR    |
| TASK-091 | PII Redaction Utility           | Redact secrets & PII fields in logs and summaries.      | Regex test set passes redaction coverage >95%.      | TASK-003     | common/redact.py             | UI-FR-056/058 |
| TASK-092 | Rate Limit & Backpressure Hooks | Basic per-IP limit on ingestion/testset/eval endpoints. | Exceed limit returns 429 with Retry-After.          | TASK-010     | services/common/ratelimit.py | Stability     |

### 5.10 QA, Testing & Hardening
| ID       | Title                        | Description                                                 | Acceptance Criteria                       | Dependencies       | Artifacts                | Req Mapping            |
|----------|------------------------------|-------------------------------------------------------------|-------------------------------------------|--------------------|--------------------------|------------------------|
| TASK-100 | Unit Test Coverage ≥70% Core | Add tests for ingestion, processing, testset, eval metrics. | Coverage report >=70% statements.         | Phases 1–4 tasks   | tests/*                  | Quality gate           |
| TASK-101 | E2E Pipeline Smoke           | Script runs minimal doc→report end-to-end.                  | Report generated & artifact chain exists. | TASK-044           | scripts/e2e_smoke.sh     | Traceability objective |
| TASK-102 | Performance Baseline Capture | Benchmark ingestion→eval latency small dataset.             | Baseline JSON stored in repo.             | TASK-101           | benchmarks/baseline.json | NFR performance        |
| TASK-103 | Load Test (Selective)        | k6 or Locust for processing & eval concurrency.             | Report p95 < target thresholds.           | TASK-015, TASK-034 | load/                    | NFR scalability        |
| TASK-104 | Resilience Chaos Drill       | Simulate transient failures (network, embedding).           | System recovers; no data corruption.      | TASK-016, TASK-033 | chaos/plan.md            | Reliability            |

### 5.11 Documentation & Operational Readiness
| ID       | Title                                 | Description                                                      | Acceptance Criteria                               | Dependencies       | Artifacts                                                                 | Req Mapping   |
|----------|---------------------------------------|------------------------------------------------------------------|---------------------------------------------------|--------------------|---------------------------------------------------------------------------|---------------|
| TASK-110 | OpenAPI Spec Drafts                   | Generate & publish service specs.                                | openapi.json for each svc committed.              | TASK-010+          | services/*/openapi.json                                                   | DevEx         |
| TASK-111 | Runbooks (Ingestion/Processing/Eval)  | Create operational runbooks (alerts, dashboards).                | Markdown runbooks link to metrics & logs queries. | TASK-080           | docs/runbooks/*.md                                                        | Ops readiness |
| TASK-112 | Deployment Manifests (K8s)            | Helm charts or k8s yaml for all services.                        | helm install dry-run success.                     | TASK-001           | deploy/helm/*                                                             | Deployment    |
| TASK-113 | ADR Finalization                      | Populate ADR-001..004 docs.                                      | ADRs merged & referenced from design.             | Existing decisions | docs/adr/*.md                                                             | Governance    |
| TASK-114 | UI Developer Guide Update             | Extend portal README with lifecycle module usage.                | README section added & lint passes.               | TASK-017           | insights-portal/README.md                                                 | DevEx         |
| TASK-115 | ADR Expansion 005-006                 | Add Telemetry taxonomy & Event schema versioning ADRs bilingual. | ADR-005 & ADR-006 exist & referenced in designs.  | TASK-113           | docs/adr/ADR-005*, ADR-006*                                               | Governance    |
| TASK-116 | Chinese UI ADR Cross-link             | Insert ADR reference table into design.ui.zh.md.                 | Table lists ADR-001..006 with status parity.      | TASK-115           | design.ui.zh.md                                                           | Governance    |
| TASK-117 | Promote ADRs to Accepted              | Change ADR-001..006 status Draft→Accepted post review.           | All ADR files show Status: Accepted.              | TASK-116           | docs/adr/ADR-00*.md                                                       | Governance    |
| TASK-118 | Event Schema Registry & CI Validation | Create registry json + validation script + CI job.               | registry + script reject unknown/changed hash.    | TASK-084           | events/schema_registry.json, scripts/validate_event_schemas.py            | Reliability   |
| TASK-119 | Telemetry Taxonomy JSON & Linter      | Machine-readable taxonomy + validation script.                   | telemetry_taxonomy.json + script pass sample run. | TASK-081           | telemetry/telemetry_taxonomy.json, scripts/validate_telemetry_taxonomy.py | Observability |

```yaml
# TASK-110 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-111 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-112 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-113 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-114 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 6
# TASK-115 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-116 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 6
# TASK-117 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-118 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-119 Governance
governance:
	status: Planned
	engineer: E3
	target_sprint: 6
```

### 5.12 Containerization & Deployment Enhancements
| ID       | Title                                       | Description                                                                                                                             | Acceptance Criteria                                                                                                                                                                       | Dependencies       | Artifacts                                                                           | Req Mapping / Rationale               |
|----------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------------------------------------------------------------------------|---------------------------------------|
| TASK-120 | Multi-Service Compose Baseline              | Split current single container into distinct services (ingestion, processing, testset, eval, reporting, ws, kg) F-compose.services.yml. | docker-compose.services.yml starts all services; healthcheck passes; logs separated; shared network & env file applied.                                                                   | TASK-001, TASK-070 | docker-compose.services.yml                                                         | Deployment scalability, ADR-001       |
| TASK-121 | Dev Hot-Reload & Source Bind Mount          | Add dev override file enabling bind mount of src + uvicorn --reload for Python services.                                                | docker-compose.dev.override.yml enables code change reflected <3s in container; documented in DOCKER_README.                                                                              | TASK-120           | docker-compose.dev.override.yml, docs update                                        | DevEx, Faster iteration               |
| TASK-122 | Image Versioning & Tag Strategy             | Introduce semantic + git SHA tagging script + Make targets (build:tag).                                                                 | make build-tag produces :vX.Y.Z and :git-<sha>; tags listed via docker images; documented strategy section added.                                                                         | TASK-120           | scripts/tag_image.sh, Makefile update, docs/deployment_guide.md                     | Traceability, Release governance      |
| TASK-123 | CI Governance & Image Build Workflow        | GitHub Actions workflow running validators (schemas, taxonomy) then build & push tagged images on main branch.                          | .github/workflows/build-governance.yml passes in PR; failing validator blocks build; pushed image appears in registry log.                                                                | TASK-118, TASK-119 | .github/workflows/build-governance.yml                                              | Automation, Reliability, Governance   |
| TASK-124 | Security & Vulnerability Scan Integration   | Integrate Trivy (or equivalent) scan into CI; fail on HIGH/CRITICAL; produce sarif artifact.                                            | CI run shows scan step; intentionally injected vulnerable image triggers failure; sarif uploaded as artifact.                                                                             | TASK-123           | .github/workflows/build-governance.yml (extended), docs/security.md                 | Security NFR                          |
| TASK-125 | Base Image Hardening & Non-Root Enforcement | Update Dockerfile: non-root user, minimize layers, add pip --no-cache-dir, network-aware pip mirror build args, separate build arg for models cache volume. | docker history shows reduced layers; container runs as non-root (id != 0); package manager caches removed; final layer count <12; offline builds log the PyPI skip guard; size reduction documented; hardening checklist present. | TASK-120           | Dockerfile (updated), docs/deployment_guide.md                                      | Security, Performance                 |
| TASK-126 | Extension / Plugin Mount Pattern            | Provide extensions/ directory volume + loader searching entrypoints allowing drop-in metrics or builders.                               | Adding sample plugin in extensions/ loads without rebuild; README section added; unit test enumerates plugin discovery.                                                                   | TASK-032, TASK-120 | extensions/sample_metric.py, services/common/plugin_loader.py                       | Extensibility, ADR-001 modularity     |
| TASK-127 | Helm Chart Decomposition                    | Create Helm chart with subcharts or templates per service + values toggles (enable/disable KG, WS).                                     | helm template succeeds; toggling kg.enabled=false excludes KG deployment; README documents values.                                                                                        | TASK-120           | deploy/helm/Chart.yaml, deploy/helm/templates/*                                     | K8s readiness, Deployment flexibility |
| TASK-128 | Health & Readiness Probes Standardization   | Add /healthz /readyz endpoints & compose/helm probe config; include startup probe for heavy init (embeddings).                          | All services respond 200; failing injected test causes non-ready status; probes defined in compose & Helm.                                                                                | TASK-120, TASK-127 | services/*/health.py, helm templates updates, compose updates                       | Reliability, Ops                      |
| TASK-129 | Kubernetes Horizontal Scaling Policies      | Provide HPA examples (CPU + custom request rate metric) for stateless services (ingestion, processing, eval).                           | Example hpa.yaml applies; scaling event observed in dry-run metrics mock; docs include tuning guidelines.                                                                                 | TASK-127, TASK-080 | deploy/helm/templates/hpa.yaml, docs/scaling.md                                     | Scalability NFR                       |
| TASK-130 | SBOM & Image Signing Pipeline               | Generate SBOM (syft) and integrate optional cosign signing with provenance attestation in CI.                                           | build-governance workflow produces CycloneDX v1.5 SBOM at sbom/sbom-main.json + signature (if key provided); verification step passes; artifacts archived.                                | TASK-123, TASK-124 | .github/workflows/build-governance.yml, sbom/sbom-main.json, docs/security.md       | Supply chain integrity                |
| TASK-131 | Optional GPU Build & Runtime Profiles       | Introduce ENABLE_GPU build arg + compose/Helm profile; expose gpu_enabled metric + doc fallback semantics.                              | GPU profile build succeeds w/out altering CPU path; metric gpu_enabled exposed; docs list enable steps & caveats.                                                                         | TASK-125           | Dockerfile (ARG), docs/deployment_guide.md, helm values                             | Performance optionality               |
| TASK-132 | Dev/CI Environment Parity Validation Script | Script compares Python version, dependency lock, extension hashes between local & container; CI gate optional.                          | Script exits non-zero on drift; sample drift test included; referenced in design §21.12.                                                                                                  | TASK-120           | scripts/validate_dev_parity.py, docs/deployment_guide.md                            | Reproducibility, DevEx                |
| TASK-133 | Policy as Code (OPA)                        | OPA Rego policies validate event schema & metrics naming (prefix/style/reserved words) in CI gate.                                      | policy/*.rego present; failing naming example test fails; CI workflow includes policy validation step.                                                                                    | TASK-118, TASK-032 | policy/*.rego, scripts/validate_policies.sh, .github/workflows/build-governance.yml | Governance, Consistency               |
| TASK-134 | Secrets Scan Gate (gitleaks)                | Integrate gitleaks to scan new commits/PRs for hardcoded secrets; fail on detection with allowlist capability.                          | CI shows gitleaks step; injected test secret triggers failure; allowlist documented.                                                                                                      | TASK-123           | .github/workflows/build-governance.yml, .gitleaks.toml, docs/security.md            | Security, Supply chain control        |

```yaml
# TASK-121 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 5
	completed_on: 2025-09-25
# TASK-122 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 5
	completed_on: 2025-09-25
# TASK-123 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 5
	completed_on: 2025-09-25
```yaml
# TASK-121 Governance
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 1p
	risk: "Lack of hot reload slows iteration"
	mitigation: "Override file with bind mount + reload"
	adr_impact: ["ADR-001"]
	ci_gate: []
	artifacts:
		- docker-compose.dev.override.yml
		- docs/DOCKER_README.md
		- docs/deployment_guide.md
	dod:
		- Dev override keeps bind mount alongside models cache volume
		- Hot reload command documented with override merge instructions
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml config
		- Uvicorn --reload commands defined
	engineer: E1
	target_sprint: 5

# TASK-122 Governance
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 1p
	risk: "Inconsistent image tagging reduces traceability"
	mitigation: "Unified script + Make targets"
	adr_impact: ["ADR-004"]
	ci_gate: []
	artifacts:
		- scripts/tag_image.sh
		- Makefile
		- VERSION
		- docs/deployment_guide.md
	dod:
		- Tag script emits v<version> and git-<sha> with optional dry-run
		- VERSION file remains single source of semantic tag
		- Deployment guide documents build-tag workflow
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 DRY_RUN=1 make tag
	engineer: E1
	target_sprint: 5
```
# TASK-124 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-125 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-126 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 5
	completed_on: 2025-09-26
# TASK-127 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-128 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-129 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-130 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-131 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-132 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-133 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-134 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-090 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-091 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-092 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
# TASK-100 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 5
# TASK-101 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 5
# TASK-102 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 5
# TASK-103 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 5
# TASK-104 Governance
governance:
	status: Planned
	engineer: E2
	target_sprint: 5
```yaml
# TASK-120 Governance
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	risk: "Service split introduces config drift"
	mitigation: "Compose validation script + health matrix test"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	artifacts:
		- docker-compose.services.yml
		- .env.compose
		- scripts/validate_compose.py
		- docs/deployment_guide.md
	dod:
		- docker compose config succeeds with default env template
		- Health endpoints scaffolded (/health)
		- Deployment guide documents env override
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 docker compose -f docker-compose.services.yml config
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-123 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-124 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
```yaml
# TASK-125 Governance
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
```
```yaml
# TASK-126 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-127 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-128 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-129 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 5
```
```yaml
# TASK-130 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
```yaml
# TASK-131 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
```yaml
# TASK-132 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
```yaml
# TASK-133 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
```yaml
# TASK-134 Governance
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```
		adr_impact: ["ADR-004"]
		ci_gate: ["unit-tests"]
		dod:
			- tag_image.sh script test
			- Version + SHA tags documented
			- Make target help updated
		engineer: E1
		target_sprint: 5
	```

# TASK-123 Governance
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	risk: "Build pipeline skips governance validations"
	mitigation: "Workflow step ordering test + fail injection"
	adr_impact: ["ADR-004","ADR-005"]
	ci_gate: ["build-governance:schemas"]
	artifacts:
		- .github/workflows/build-governance.yml
		- scripts/validate_task_status.py
		- scripts/validate_compose.py
	dod:
		- Workflow orchestrates validators before image build
		- Governance scripts fail-fast on status drift
		- Build job tags/pushes GHCR images on main
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 python3 scripts/validate_task_status.py
	engineer: E1 (E3 CI front-end budget step integration)
	target_sprint: 5

# TASK-124 Governance
governance:
	status: Verified
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "Critical vuln passes undetected"
	mitigation: "Injected CVE fixture triggers failure"
	adr_impact: ["ADR-004"]
	ci_gate: ["security-scan"]
	artifacts:
		- .github/workflows/build-governance.yml
		- docs/security.md
	dod:
		- Trivy filesystem & image scans produce SARIF artifacts
		- HIGH/CRITICAL findings fail the workflow via exit-code 1
		- Security guide documents remediation & skip policy guidance
	completed_on: 2025-09-25
	verification:
		- Workflow validation via static review (Trivy steps present)
	engineer: E1 (secops assist)
	target_sprint: 5

# TASK-125 Governance
governance:
	status: Verified
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "Unhandled base image CVEs or privilege escalation"
	mitigation: "Hardened Dockerfile with multi-stage build and scan hooks"
	adr_impact: ["ADR-004"]
	ci_gate: ["unit-tests"]
	artifacts:
		- Dockerfile
		- docker-compose.services.yml
		- .env.compose
		- docs/deployment_guide.md
		- docs/hardening_checklist.md
	dod:
		- Dockerfile refactored into builder/runtime stages; runtime contains no build toolchain packages
		- Non-root service user retained with owned `/app`, `${MODELS_CACHE_PATH}`, and `${EXTENSIONS_DIR}` directories
		- Dependency installation consolidated with `pip --no-cache-dir`, `PIP_DISABLE_PIP_VERSION_CHECK`, and bytecode suppression
		- Build-time pip mirror/offline guard configurable via `PIP_INDEX_URL`/* and documented in hardening checklist to prevent repeated timeout loops
		- Base image patched (`python:3.11-slim-bookworm`) with `apt-get upgrade` executed per stage; MODELS_CACHE build arg still surfaced with compose volume wiring
		- Hardening checklist & deployment guide updated to capture verification steps and multi-stage rationale
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 docker build -t rag-eval:test .
		- 2025-09-25 docker history rag-eval:test | head -n 12
		- 2025-09-25 grep "python:3.11-slim-bookworm" Dockerfile
		- 2025-09-25 python3 -m compileall services
	engineer: E1
	target_sprint: 5

# TASK-126 Governance
governance:
	status: Verified
	owner: platform-extensions@team
	priority: P2
	estimate: 2p
	risk: "Plugin discovery regresses without coverage"
	mitigation: "Unit tests exercising register() and fallback payloads"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	artifacts:
		- services/common/plugin_loader.py
		- extensions/sample_metric.py
		- test_plugin_loader.py
		- docs/DOCKER_README.md
		- docs/deployment_guide.md
	dod:
		- Loader discovers register() based plugins and attribute payload fallback
		- Sample plugin loads without image rebuild via mounted directory
		- Documentation explains EXTENSIONS_DIR override and usage
		- Automated test enumerates discovered plugins
	completed_on: 2025-09-26
	verification:
		- 2025-09-26 pytest test_plugin_loader.py
	engineer: E3 (E2 consult for metrics plugin alignment)
	target_sprint: 6

	```yaml
	# TASK-127 Governance
	governance:
		status: Planned
		owner: platform-deploy@team
		priority: P1
		estimate: 2p
		risk: "Monolithic chart complexity"
		mitigation: "Template linter + values toggles tests"
		adr_impact: ["ADR-004"]
		ci_gate: ["unit-tests"]
		dod:
			- helm template success test
			- kg/ws disabled diff test
			- README values table
		engineer: E1
		target_sprint: 5
	# TASK-128 Governance
	governance:
		status: Planned
		owner: platform-deploy@team
		priority: P1
		estimate: 1p
		risk: "Missing readiness hides failures"
		mitigation: "Probe failure fixture"
		adr_impact: ["ADR-004"]
		ci_gate: ["unit-tests"]
		dod:
			- /healthz test
			- /readyz test
			- Startup probe doc
		engineer: E1
		target_sprint: 5
	# TASK-129 Governance
	governance:
		status: Planned
		owner: platform-deploy@team
		priority: P2
		estimate: 1p
		risk: "Over/under scaling"
		mitigation: "HPA dry-run metrics test"
		adr_impact: []
		ci_gate: ["unit-tests"]
		dod:
			- HPA example applied test
			- Scaling event doc
			- Tuning guide
		engineer: E1
		target_sprint: 6
	# TASK-131 Governance
	governance:
		status: Planned
		owner: platform-deploy@team
		priority: P3
		estimate: 1p
		risk: "GPU path diverges from CPU"
		mitigation: "Parity build test"
		adr_impact: []
		ci_gate: ["unit-tests"]
		dod:
			- gpu_enabled metric test
			- CPU fallback test
			- README GPU section
		engineer: E1
		target_sprint: 6
	# TASK-132 Governance (Engineer Assignment)
	governance:
		engineer: E1
		target_sprint: 6
	# TASK-133 Governance
	governance:
		status: Planned
		owner: platform-governance@team
		priority: P2
		estimate: 2p
		risk: "Policy gaps allow inconsistent metrics"
		mitigation: "Negative naming test cases"
		adr_impact: ["ADR-005"]
		ci_gate: ["policy-validate"]
		dod:
			- Rego tests pass
			- Naming violation sample
			- README policy section
		engineer: E1 (E2 consult for metrics)
		target_sprint: 6
	# TASK-134 Governance
	governance:
		status: Planned
		owner: platform-secops@team
		priority: P1
		estimate: 1p
		risk: "Secrets leak to repo"
		mitigation: "Injected secret test"
		adr_impact: ["ADR-004"]
		ci_gate: ["security-scan"]
		dod:
			- gitleaks config committed
			- Failing secret test
			- Allowlist doc
		engineer: E1
		target_sprint: 5
	```
# TASK-132 Governance
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "Environment drift reduces reproducibility"
	mitigation: "Parity script diff test"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- Drift exit code test
		- JSON parity report
		- README parity section
```

```yaml
# TASK-130 Governance
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 4p
	risk: "Key management complexity or lost signing key blocks deployments"
	mitigation: "Allow unsigned fallback with clear annotation; KMS-backed key w/ least privilege"
	adr_impact: ["ADR-004"]
	ci_gate: ["security-scan","sbom-generate","image-sign"]
	sbom_format: CycloneDX-1.5
	artifacts:
		- sbom/sbom-main.json
		- sbom/sbom-diff.json
		- attest/provenance.intoto.jsonl
	slo:
		pipeline_additional_time_seconds_p95: 90
		critical_vuln_allowed: 0
	metrics:
		- supplychain_vuln_count{severity="CRITICAL"}
		- supplychain_unsigned_image_total
	logs:
		- code=SBOM_GENERATED level=INFO
		- code=IMAGE_SIGNED level=INFO
		- code=SIGNING_SKIPPED level=WARN
	dod:
		- syft produces CycloneDX JSON stored under sbom/
		- Trivy SARIF uploaded with 0 CRITICAL/HIGH (allow configurable medium tolerance)
		- cosign verify example documented in security.md
		- Missing signature triggers workflow annotation
		- Deployment guide updated with SBOM / signing section
	engineer: E1 (secops partnering)
	target_sprint: 6
```

#### TASK-126 Subtasks
| Sub-ID    | Title                          | Description                                                            | Acceptance Criteria                                                 | Dependencies | Artifacts                         | Notes            |
|-----------|--------------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------|--------------|-----------------------------------|------------------|
| TASK-126a | Directory Watch / Reload (Opt) | Optional watchdog to reload plugins on file change in dev mode.        | File add/remove reflected without restart in dev; disabled in prod. | TASK-126     | services/common/plugin_loader.py  | Dev productivity |
| TASK-126b | Sandbox & Allowlist            | Restrict imports via allowlist; sandbox exec context.                  | Disallowed import raises clear error & logs security event.         | TASK-126a    | services/common/plugin_sandbox.py | Security         |
| TASK-126c | Version Negotiation            | Read plugin manifest (YAML/JSON) with contract_version & capabilities. | Incompatible version skipped with warning.                          | TASK-126b    | extensions/manifest.schema.json   | Compatibility    |
| TASK-126d | Failure Telemetry Events       | Emit structured events (plugin.failed, plugin.loaded).                 | Events appear in logs & metrics counters update.                    | TASK-126c    | services/common/plugin_events.py  | Observability    |

```yaml
# TASK-126a Governance
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "Reload causes memory leak"
	mitigation: "Reload stress test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Add/remove file reload test
		- Prod disabled assertion
		- README reload section
# TASK-126b Governance
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P1
	estimate: 1p
	risk: "Untrusted import executes arbitrary code"
	mitigation: "Allowlist + sandbox test"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests","security-scan"]
	dod:
		- Blocked import test
		- Security event log
		- README sandbox doc
# TASK-126c Governance
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "Incompatible plugin version loaded"
	mitigation: "Version negotiation test"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Skip incompatible test
		- Warning log check
		- Manifest schema doc
# TASK-126d Governance
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "Plugin failures not observable"
	mitigation: "Event + metric tests"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Failure event test
		- Metrics increment
		- README telemetry
```

#### TASK-130 Subtasks
| Sub-ID    | Title                   | Description                                                                | Acceptance Criteria                                             | Dependencies | Artifacts                      | Notes           |
|-----------|-------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------|--------------|--------------------------------|-----------------|
| TASK-130a | SBOM Generation         | Generate CycloneDX JSON SBOM via syft for each image.                      | sbom-main.json produced; schema validates.                      | TASK-130     | sbom/sbom-main.json            | Baseline        |
| TASK-130b | Vulnerability Diff      | Compare current scan vs prior run; produce diff JSON.                      | Diff highlights new HIGH/CRITICAL; exit code reflects severity. | TASK-130a    | sbom/sbom-diff.json            | Governance gate |
| TASK-130c | Conditional Signing     | cosign sign if key present; else annotate unsigned path.                   | Signed image passes cosign verify in CI; unsigned logs warning. | TASK-130b    | attest/*.intoto.jsonl          | Supply chain    |
| TASK-130d | Attestation & Retention | Generate provenance attestation + prune old SBOMs beyond retention window. | Attestation file present; pruning keeps ≤N historical sets.     | TASK-130c    | attest/provenance.intoto.jsonl | Hygiene         |

```yaml
# TASK-130a Governance
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "SBOM schema drift"
	mitigation: "Schema validation test"
	adr_impact: ["ADR-004"]
	ci_gate: ["sbom-generate"]
	dod:
		- SBOM file exists
		- Schema validation pass
		- README SBOM section
# TASK-130b Governance
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "Vuln diff misses new criticals"
	mitigation: "Diff unit test"
	adr_impact: []
	ci_gate: ["security-scan"]
	dod:
		- New vuln detection test
		- Exit code logic test
		- Docs diff usage
# TASK-130c Governance
governance:
	status: Planned
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "Unsigned image undetected"
	mitigation: "Verify command test"
	adr_impact: []
	ci_gate: ["image-sign"]
	dod:
		- Cosign verify test
		- Warning on unsigned
		- Docs signing usage
# TASK-130d Governance
governance:
	status: Planned
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "Retention cleanup fails"
	mitigation: "Retention window test"
	adr_impact: []
	ci_gate: ["sbom-generate"]
	dod:
		- Prune test
		- Attestation present
		- README retention note
```

#### TASK-132 Subtasks
| Sub-ID    | Title                       | Description                                                       | Acceptance Criteria                                      | Dependencies | Artifacts                      | Notes           |
|-----------|-----------------------------|-------------------------------------------------------------------|----------------------------------------------------------|--------------|--------------------------------|-----------------|
| TASK-132a | Python Version Compare      | Compare local vs container Python major.minor; warn on mismatch.  | Mismatch triggers non-zero exit & JSON report field.     | TASK-132     | scripts/validate_dev_parity.py | Drift detection |
| TASK-132b | Dependency Lock Hash        | Compute hash of lock file(s); compare against container snapshot. | Changed hash produces drift flag; whitelist respected.   | TASK-132a    | scripts/validate_dev_parity.py | Integrity       |
| TASK-132c | Extension Fingerprint       | Hash each extensions/ file & compare.                             | Added/removed/changed plugin surfaces in report diff.    | TASK-132b    | scripts/validate_dev_parity.py | Extensibility   |
| TASK-132d | Drift Whitelist & Formatter | Support whitelist of acceptable drifts; output markdown & JSON.   | Report includes summary table; whitelisted drift marked. | TASK-132c    | scripts/validate_dev_parity.py | Reporting       |

```yaml
# TASK-132a Governance
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "Python version drift unnoticed"
	mitigation: "Version mismatch test"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- Version diff test
		- JSON report present
		- README parity section
# TASK-132b Governance
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "Lock file drift not flagged"
	mitigation: "Hash compare test"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- Drift flag test
		- Allowlist doc
		- Report diff section
# TASK-132c Governance
governance:
	status: Planned
	owner: platform-parity@team
	priority: P2
	estimate: 1p
	risk: "Extension changes untracked"
	mitigation: "Fingerprint diff test"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- Fingerprint report
		- Diff test
		- README fingerprint doc
# TASK-132d Governance
governance:
	status: Planned
	owner: platform-parity@team
	priority: P2
	estimate: 1p
	risk: "Whitelist hides real drift"
	mitigation: "Whitelist override test"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- Whitelist annotation
		- MD + JSON outputs
		- README whitelist policy
```

## 6. Requirement Coverage Summary
High-level mapping (detail per task table):
- Ingestion & Processing FR: TASK-010..016
- Testset FR-013~016: TASK-020..024
- Evaluation FR-017~022: TASK-030..035
- Reporting FR-037~040: TASK-041..044
- KM Export FR-041/042: TASK-045..046
- UI-FR-016~018 (KG): TASK-060..065
- UI Realtime & Performance (UI-FR-049~055): TASK-070..073, TASK-082, TASK-084
- Privacy & Redaction (UI-FR-056/058): TASK-091
- Feature Flags & Lazy Loading (UI-NFR-006): TASK-065, TASK-082
- Traceability Goal (SMART #4): Artifact chain tasks (010..016, 020..024, 030..035, 041..044, 045) + TASK-083 manifest.

## 7. Risks & Mitigations (Extract)
| Risk                                | Impact           | Mitigation Task                                      |
|-------------------------------------|------------------|------------------------------------------------------|
| Large KG slows UI                   | Perf degradation | TASK-065 sampling cap, TASK-066 subgraph             |
| Event drift / missed updates        | Stale UI state   | TASK-071 sequencing + TASK-072 resync                |
| Duplicate artifacts inflate storage | Cost & confusion | TASK-015 idempotent processing, TASK-020 config hash |
| Metric regression unnoticed         | Quality drift    | TASK-082 CI, TASK-102 baseline                       |
| Unbounded subgraph requests         | Backend load     | TASK-066 rate limits + deterministic sampling        |

## 8. Optimization & Enhancement Recommendations
- Manifest Integrity (TASK-083) escalate to signed manifest v2 after baseline stable.
- Schema Registry: Extend TASK-084 with JSON Schema hash pinning & CI drift detection.
- Metrics Caching Layer: Use evaluation determinism hash to memoize kpis.json for identical runs.
- Progressive WebSocket Adoption: Keep polling fallback during burn-in (TASK-073) until error rate < 0.5%.
- Persona/Scenario Drill-down: Post-M3 analytics UI to surface cross-run persona coverage variance.

## 9. RACI Snapshot (Abbreviated)
| Role                   | Responsible (R) | Accountable (A) | Consulted (C) | Informed (I) |
|------------------------|-----------------|-----------------|---------------|--------------|
| Service Implementation | Platform Eng    | Platform Lead   | QA, Security  | Stakeholders |
| UI Lifecycle Module    | Frontend Eng    | Frontend Lead   | Platform Eng  | Stakeholders |
| KG Feature             | Data Eng        | Platform Lead   | Frontend Eng  | Stakeholders |
| Telemetry & WS         | Platform Eng    | Platform Lead   | SRE           | Stakeholders |
| Documentation & ADR    | Tech Writer     | Platform Lead   | Eng Leads     | Org          |

## 10. Acceptance & Change Control
Changes to scope require updating this plan (section version bump) and linking to corresponding ADR if architectural. Minor task ordering changes do not require version bump but must retain requirement coverage.

## 12. Subtask → Parent Traceability Matrix

This section consolidates all defined subtasks (a–d) under high-impact parent tasks for governance, scheduling clarity, and dependency risk review.

### 12.1 Matrix (English)
| Subtask   | Parent   | Domain       | Focus                            | Key Artifact                                | Depends On  | Downstream / Consumer        | Key Acceptance Signal       |
|-----------|----------|--------------|----------------------------------|---------------------------------------------|-------------|------------------------------|-----------------------------|
| TASK-015a | TASK-015 | Processing   | Tokenization & boundaries        | processing/stages/tokenizer.py              | Parent init | 015b/015c                    | Stable spans                |
| TASK-015b | TASK-015 | Processing   | Chunk size/overlap rules         | processing/stages/chunk_rules.py            | 015a        | 015c/015d                    | Size & determinism          |
| TASK-015c | TASK-015 | Processing   | Embedding batch exec + retries   | processing/stages/embed_executor.py         | 015b        | 015d, evaluation             | Retry & breaker metrics     |
| TASK-015d | TASK-015 | Processing   | Persistence & integrity manifest | processing/stages/chunk_persist.py          | 015c        | Testset/Eval stages          | Hash/count parity           |
| TASK-032a | TASK-032 | Eval         | Plugin interface contract        | eval/metrics/interface.py                   | Parent init | 032b/032c, 126c              | Clear missing-method errors |
| TASK-032b | TASK-032 | Eval         | Baseline metrics impl            | eval/metrics/baseline/*.py                  | 032a        | Evaluation run (030–035)     | Deterministic scores        |
| TASK-032c | TASK-032 | Eval         | Discovery + failure isolation    | eval/metrics/loader.py                      | 032b        | Future plugins, 126*         | Faulty plugin isolation     |
| TASK-062a | TASK-062 | KG           | Node property enrichment         | kg/extract.py                               | Parent init | 062b/062c/summary            | Properties completeness     |
| TASK-062b | TASK-062 | KG           | Jaccard & Overlap relations      | kg/relationships.py                         | 062a        | 062c/062d                    | >0 relationships sample     |
| TASK-062c | TASK-062 | KG           | Cosine similarity + fallback     | kg/relationships.py                         | 062b        | 062d tuning                  | Graceful no-embed skip      |
| TASK-062d | TASK-062 | KG           | Threshold tuning harness         | scripts/kg_threshold_tune.py                | 062c        | KG ops / threshold decisions | JSON metrics output         |
| TASK-126a | TASK-126 | Extensions   | Dev reload (optional)            | services/common/plugin_loader.py            | Parent init | 126b/126c                    | Live reload works           |
| TASK-126b | TASK-126 | Extensions   | Sandbox & allowlist              | services/common/plugin_sandbox.py           | 126a        | 126c/126d                    | Blocked disallowed import   |
| TASK-126c | TASK-126 | Extensions   | Version negotiation              | extensions/manifest.schema.json             | 126b        | 032a interface alignment     | Incompatible skipped        |
| TASK-126d | TASK-126 | Extensions   | Failure telemetry events         | services/common/plugin_events.py            | 126c        | Observability / SRE          | Events & counters present   |
| TASK-130a | TASK-130 | Supply Chain | SBOM generation                  | sbom/sbom-main.json                         | Parent init | 130b/130d                    | Valid CycloneDX             |
| TASK-130b | TASK-130 | Supply Chain | Vulnerability diff               | sbom/sbom-diff.json                         | 130a        | 130c signing decision        | New HIGH/CRITICAL flagged   |
| TASK-130c | TASK-130 | Supply Chain | Conditional signing              | attest/provenance.intoto.jsonl & signatures | 130b        | Deployment pipeline          | Verified or warning logged  |
| TASK-130d | TASK-130 | Supply Chain | Attestation & retention          | attest/provenance.intoto.jsonl              | 130c        | Audit & compliance           | Old artifacts pruned        |
| TASK-132a | TASK-132 | Parity       | Python version compare           | scripts/validate_dev_parity.py              | Parent init | 132b/132c/132d               | Exit code on mismatch       |
| TASK-132b | TASK-132 | Parity       | Lock file hash compare           | scripts/validate_dev_parity.py              | 132a        | 132d                         | Drift flag set              |
| TASK-132c | TASK-132 | Parity       | Extension fingerprint            | scripts/validate_dev_parity.py              | 132b        | 132d                         | Diff enumerated             |
| TASK-132d | TASK-132 | Parity       | Whitelist & formatted report     | scripts/validate_dev_parity.py              | 132c        | CI gate / reviewers          | Whitelisted drifts tagged   |

### 12.2 Matrix (Chinese)
| 子任務    | 主任務   | 功能領域     | 核心焦點            | 主要產出/檔案                       | 直接依賴   | 下游/被誰使用     | 驗收核心指標    |
|-----------|----------|--------------|---------------------|-------------------------------------|------------|-------------------|-----------------|
| TASK-015a | TASK-015 | Processing   | Tokenizer + 邊界    | processing/stages/tokenizer.py      | 主任務啟動 | 015b/015c         | 邊界穩定        |
| TASK-015b | TASK-015 | Processing   | Chunk 規則/重疊     | processing/stages/chunk_rules.py    | 015a       | 015c/015d         | 無超限 & 決定性 |
| TASK-015c | TASK-015 | Processing   | 嵌入批次 + 重試     | processing/stages/embed_executor.py | 015b       | 015d, 評估        | 失敗/斷路指標   |
| TASK-015d | TASK-015 | Processing   | 持久化 + Manifest   | processing/stages/chunk_persist.py  | 015c       | Testset/Eval      | 雜湊/計數一致   |
| TASK-032a | TASK-032 | Eval         | 外掛介面契約        | eval/metrics/interface.py           | 主任務啟動 | 032b/032c, 126c   | 缺方法即錯      |
| TASK-032b | TASK-032 | Eval         | 基礎指標實作        | eval/metrics/baseline/*.py          | 032a       | 評估流程          | 決定性分數      |
| TASK-032c | TASK-032 | Eval         | 探索 + 隔離         | eval/metrics/loader.py              | 032b       | 未來外掛, 126*    | 壞外掛不連鎖    |
| TASK-062a | TASK-062 | KG           | 節點屬性增豐        | kg/extract.py                       | 主任務啟動 | 062b/062c/summary | 屬性完整率      |
| TASK-062b | TASK-062 | KG           | Jaccard & Overlap   | kg/relationships.py                 | 062a       | 062c/062d         | >0 關係樣本     |
| TASK-062c | TASK-062 | KG           | Cosine + fallback   | kg/relationships.py                 | 062b       | 062d              | 無向量仍通過    |
| TASK-062d | TASK-062 | KG           | 閾值調參工具        | scripts/kg_threshold_tune.py        | 062c       | KG 連線/閾值決策  | JSON 報告       |
| TASK-126a | TASK-126 | Extensions   | Dev 目錄熱重載      | services/common/plugin_loader.py    | 主任務啟動 | 126b/126c         | 變更即載入      |
| TASK-126b | TASK-126 | Extensions   | Sandbox Allowlist   | services/common/plugin_sandbox.py   | 126a       | 126c/126d         | 禁匯入阻擋      |
| TASK-126c | TASK-126 | Extensions   | 版本協商            | extensions/manifest.schema.json     | 126b       | 032a 介面對齊     | 不相容跳過      |
| TASK-126d | TASK-126 | Extensions   | 失敗 telemetry 事件 | services/common/plugin_events.py    | 126c       | 可觀測/SRE        | 事件+計數       |
| TASK-130a | TASK-130 | Supply Chain | SBOM 生成           | sbom/sbom-main.json                 | 主任務啟動 | 130b/130d         | CycloneDX 驗證  |
| TASK-130b | TASK-130 | Supply Chain | 漏洞差異            | sbom/sbom-diff.json                 | 130a       | 130c 簽章         | 新高風險標示    |
| TASK-130c | TASK-130 | Supply Chain | 條件簽章            | attest/provenance.intoto.jsonl 等   | 130b       | 部署流程          | 簽章驗證/警告   |
| TASK-130d | TASK-130 | Supply Chain | Attestation + 保留  | attest/provenance.intoto.jsonl      | 130c       | 稽核/合規         | 舊檔清理        |
| TASK-132a | TASK-132 | Parity       | Python 版本比對     | scripts/validate_dev_parity.py      | 主任務啟動 | 132b/132c/132d    | 不一致退出碼    |
| TASK-132b | TASK-132 | Parity       | 依賴鎖雜湊          | scripts/validate_dev_parity.py      | 132a       | 132d              | 漂移標誌        |
| TASK-132c | TASK-132 | Parity       | 擴充指紋            | scripts/validate_dev_parity.py      | 132b       | 132d              | 差異列出        |
| TASK-132d | TASK-132 | Parity       | 白名單 & 報告       | scripts/validate_dev_parity.py      | 132c       | CI Gate / 審閱    | 白名單標註      |

### 12.3 Observations
Most dependent chain: TASK-015 (feeds evaluation & testset). Supply chain risk path: 130a→130b→130c→130d. Extension safety chain: 126a→126b→126c→126d (touches 032a contract). KG quality tuning loop: 062a→062b→062c→062d.

### 12.4 Potential Automation
1. Script to parse `tasks*.md` and regenerate this matrix (Markdown + JSON) ensuring drift detection.
2. CI gate: validate every listed subtask's parent exists and governance skeleton present.
3. Coverage metric: percentage of high-impact parents with decomposed subtasks (currently 100% for selected set).

## 11. Task Field & Governance Specification
To improve consistency, traceability, and automation, high-risk or core tasks SHOULD append a `governance` YAML block. Field definitions:

| Field                   | Meaning                        | Guidance                                                 |
|-------------------------|--------------------------------|----------------------------------------------------------|
| status                  | Task lifecycle state           | Planned / In-Progress / Blocked / Done / Verified        |
| owner                   | Responsible individual or team | Use group email or GitHub team slug                      |
| priority                | Execution priority             | P0 (critical) .. P3 (low)                                |
| estimate                | Effort size                    | Story Points (team baseline) or person-days              |
| risk                    | Principal risk statement       | One concise sentence incl. trigger                       |
| mitigation              | How risk impact is reduced     | Degrade path / fallback / circuit breaker                |
| adr_impact              | Related ADRs affected          | List ADR IDs requiring update/reference                  |
| ci_gate                 | CI jobs gating merge           | Workflow job names enforcing quality                     |
| slo                     | Quantitative targets           | Latency / error rate / init time etc.                    |
| metrics                 | New metrics to expose          | snake_case names; histograms end with _seconds or _bytes |
| logs                    | Key log events                 | Use code=PREFIX pattern for searchability                |
| artifacts               | Additional artifacts           | Items not already shown in Artifacts column              |
| plugin_contract_version | (Plugins) API version          | Increment on breaking interface change                   |
| failure_isolation       | (Plugins) Isolation strategy   | e.g. per-plugin try/except & noop fallback               |
| sbom_format             | (Supply chain) SBOM format     | CycloneDX / SPDX                                         |
| dod                     | Definition of Done checklist   | Consistent acceptance gating                             |

Recommended baseline DoD checklist:
1. Tests: unit + error paths; integration if external I/O.
2. Observability: metrics + key logs + trace tags where applicable.
3. Documentation: README / design / ADR updates (if architectural change).
4. Security & compliance: no new HIGH/CRITICAL CVEs; secrets not committed.
5. Performance: ≤120% of prior baseline latency (if performance-sensitive).
6. Degradation: feature failure does not crash core pipeline; emits structured error event.

Automation suggestion: `scripts/validate_tasks.py` can parse all governance blocks and enforce required fields & naming conventions. Future CI gate can reject PRs missing mandatory governance on P0/P1 tasks.

### 11.1 Quick Reference (Required / Conditional Fields)
| Field                   | Required               | Conditional Logic                                   |
|-------------------------|------------------------|-----------------------------------------------------|
| status                  | Yes                    | Always present                                      |
| owner                   | Yes                    | Team or individual                                  |
| priority                | Yes                    | P0 for critical path gates                          |
| estimate                | Yes                    | Story Points (team baseline)                        |
| risk                    | Yes                    | At least one risk for P0/P1                         |
| mitigation              | Yes                    | Mirrors risk entries                                |
| adr_impact              | Yes                    | Empty list if none                                  |
| ci_gate                 | Optional               | Required if task introduces validator or build step |
| slo                     | Optional               | Add for latency / reliability sensitive tasks       |
| metrics                 | Yes (if service logic) | Skip only for pure docs tasks                       |
| logs                    | Yes (runtime tasks)    | At least one structured event code=...              |
| artifacts               | Optional               | List extra outputs not in main table                |
| plugin_contract_version | Conditional            | Only for plugin or extension systems                |
| failure_isolation       | Conditional            | Required for plugin/extension execution paths       |
| sbom_format             | Conditional            | Supply chain tasks (e.g. TASK-130)                  |
| dod                     | Yes                    | Minimum shared checklist items                      |

### 11.2 YAML Skeleton Template
```yaml
governance:
	status: Planned
	owner: <team-or-user>
	priority: P1
	estimate: 3p            # Story Points
	risk: "<primary risk>"
	mitigation: "<fallback / circuit breaker>"
	adr_impact: []          # e.g. ["ADR-001"]
	ci_gate: ["unit-tests"]
	slo:
		<metric_name>: <target>
	metrics:
		- <service_operation_duration_seconds>
	logs:
		- code=<EVENT_CODE> level=INFO
	artifacts: []
	plugin_contract_version: 1
	failure_isolation: "try-except per plugin"
	sbom_format: CycloneDX-1.5
	dod:
		- Tests added (unit + error paths)
		- Metrics exported
		- Key logs present
		- Docs updated (README/design/ADR)
		- No new HIGH/CRITICAL CVEs
		- Graceful degradation path verified
```

### 11.3 CI Gate Naming Conventions
| Purpose                      | Suggested Job Name       |
|------------------------------|--------------------------|
| Unit & Lint                  | unit-tests               |
| Coverage Gate                | coverage-check           |
| Schema & Taxonomy Validation | build-governance:schemas |
| Bundle Size Budget           | bundle-size-guard        |
| Security Scan                | security-scan            |
| SBOM Generation              | sbom-generate            |
| Image Signing                | image-sign               |
| Performance Smoke            | perf-baseline            |
| Dev/CI Parity                | parity-validate          |

### 11.4 Status Workflow
Planned → In-Progress → (Blocked ↔ In-Progress) → Done → Verified (post-review or automated evidence).

Blocked tasks MUST include a short cause + next action in PR description or tracking issue.

---
End of document.