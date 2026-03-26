# Release Notes — v1.0.0

> **Branch:** `feat/graph-context-relevance` → `main`
> **Date:** 2026-03-27
> **Commits in this release:** 226
> **Diff summary:** 724 files changed · 139,971 insertions · 276 deletions

---

## Executive Summary

Version 1.0.0 marks the **first production-ready release** of the Domain-Specific LLM Evaluation
Pipeline. This release is the result of a comprehensive architectural overhaul that transforms an
exploratory research prototype into a fully containerised, microservice-based evaluation platform.

The headline contribution is the **Graph Context Relevance (GCR)** evaluator — a novel,
fully-offline metric that measures the topological quality of retrieved context using graph theory.
This is complemented by an **LLM-powered Auto-Insights engine**, an **interactive QA Debugger**
with entity highlighting, a modern **Insights Portal** UI, and a **Zero Host Dependency** DevOps
posture underpinned by a 733-test parallel test suite.

---

## ✨ Major Features

### 1. Graph Context Relevance Evaluator (`eval-pipeline/src/evaluation/graph_context_relevance.py`)

The flagship algorithmic contribution of this release. The `GraphContextRelevanceEvaluator`
computes a composite score across three complementary graph-topological dimensions:

$$
\mathrm{GCR} = \mathrm{clip}(\alpha \cdot S_e + \beta \cdot S_c - \gamma \cdot P_h,\ 0.0,\ 1.0)
$$

| Component | Symbol | Definition | Default weight |
|-----------|--------|------------|----------------|
| **Entity Overlap** | $S_e\ [0,1]$ | Jaccard similarity between question/answer token sets and per-node keyphrases + content. Measures semantic relevance of the retrieved subgraph. | α = 0.4 |
| **Structural Connectivity** | $S_c\ [0,1]$ | Ratio of the largest connected component to total retrieved nodes, computed over an undirected graph projection. Rewards coherent retrieval; penalises fragmented results. | β = 0.4 |
| **Hub Noise Penalty** | $P_h\ [0,1]$ | Fraction of retrieved nodes whose degree exceeds μ + 2σ across the full graph. Discourages retrieval of massive "hub" nodes that dilute context quality. | γ = 0.2 |

**Key properties:**
- 100% offline and deterministic — zero model calls, zero network I/O.
- Complexity: O(N + E) graph build; O(|R| · |Q|) entity overlap; O(|R| + |E_R|) connectivity.
- Fully configurable weights via `GraphContextRelevanceConfig`.
- Graceful fallback to neutral scores when the graph store is unavailable.

### 2. Content-Hash-Addressed SQLite GraphStore (`eval-pipeline/src/utils/graph_store.py`)

A persistent, thread-safe graph caching layer built on SQLite:

- **Content-hash addressing:** graphs are keyed by SHA-256 digest of the source document
  corpus, ensuring deterministic cache hits across pipeline runs without manual invalidation.
- **NetworkX round-trip:** serialises and deserialises full `networkx.Graph` objects including
  all node/edge attributes (entities, keyphrases, embeddings, summaries).
- **EvaluationDispatcher facade** (`eval-pipeline/src/evaluation/evaluation_dispatcher.py`):
  unified metric routing for `TestsetSpec` objects — dispatches GCR, RAGAS, contextual keyword,
  and hybrid evaluators through a single interface.
- CI isolation mode: deterministic offline fallbacks for all metrics when live models are absent.

### 3. LLM-Powered Auto-Insights Engine

A zero-configuration AI narrative layer for evaluation results:

- **Backend:** `POST /api/v1/insights/generate` on the `reporting` service (port 8005).
  - Accepts KPIs, verdict, failing metrics, and threshold deltas.
  - Constructs a structured prompt that teaches the LLM metric semantics:
    $S_e$ (entity overlap), $S_c$ (structural connectivity), $P_h$ (hub noise penalty,
    inverted), Faithfulness, and AnswerRelevancy.
  - OpenAI-compatible async client — works with GPT-4o-mini, Azure OpenAI, vLLM, and Ollama
    via `INSIGHTS_LLM_MODEL` and `INSIGHTS_API_BASE_URL` environment variables.
  - Hardened error surface: 503 on missing API key (actionable message), 502 on upstream LLM
    errors, separate `AuthenticationError` handling.
  - CORS middleware enabled for cross-origin browser fetch.
- **Frontend:** `AiInsightCard` React component.
  - Premium animated gradient border (CSS keyframe) during generation.
  - Sparkle icon with pulse animation.
  - Renders LLM markdown response inline — no external markdown dependencies.

### 4. Interactive QA Debugger with Entity Highlighting (`insights-portal/src/app/routes/QAFailureExplorer.tsx`)

A full master-detail debugging interface for QA failure analysis:

- **Master panel:** paginated, filterable list of all QA pairs with per-row metric badges
  (GCR score, faithfulness, answer relevancy).
- **Detail panel:** side-by-side display of question, expected answer, generated answer, and
  full retrieved context chunks with inline diff highlighting.
- **Entity highlighting** (`insights-portal/src/utils/textHighlighter.ts`): automatically
  extracts and highlights shared entities between the question and each context passage,
  making it immediately visible why a retrieval succeeded or failed.
- Executive Overview integration: drill-down links from the summary KPI cards directly into
  the QA Debugger for failing metric buckets.
- 1,086 net lines added; replaced a 180-line placeholder with a production-grade component.

### 5. Revamped Insights Portal Dashboard (`insights-portal/`)

End-to-end modernisation of the frontend evaluation console:

- **Modern design system:** new CSS custom-property theme (`theme.css`, 330 lines) with dark
  mode, responsive layouts, and a consistent component library.
- **Cytoscape KG Visualiser:** interactive knowledge-graph explorer with force-directed layout,
  node-type colouring, and edge-weight display (`CytoscapeGraph.tsx`).
- **Analytics & Compare views:** multi-run overlays, box/scatter charts with brush-to-filter,
  baseline comparison with mean/median/p50/p90 deltas, and CSV/XLSX export with branding.
- **PDF export service:** generates stamped evaluation reports from the portal.
- **Accessibility & i18n:** RTL smoke tests, a11y audits, and full internationalisation
  (`index.ts` locale registry).
- **QA Bookmarks & SLA sliders:** persistent user preferences via `localStorage`.

---

## 🏗️ Backend Services Architecture

This release establishes an 11-service Docker Compose topology:

| Service | Port | Role |
|---------|------|------|
| `kg` | 8001 | Knowledge Graph builder and query API |
| `ingestion` | 8002 | Document submission with SQLite persistence and KM checksum deduplication |
| `testset` | 8003 | RAGAS testset generation |
| `eval` | 8004 | Metric evaluation (GCR, RAGAS, hybrid, contextual keyword) |
| `reporting` | 8005 | Aggregation, artefact storage, and AI Insights generation |
| `insights-portal` | 8006 | React SPA (served by Nginx) |
| `processing` | 8007 | Async document processing workers |
| `webhook` | 8008 | Git webhook daemon for CI-triggered pipeline runs |
| `adapter` | 8009 | Backend adapter for external LLM providers |
| `minio` | 9000/9001 | S3-compatible artefact store |
| `minio-init` | — | One-shot bucket provisioning |

---

## 🚀 DevOps & CI/CD

### Zero Host Dependency Initiative (Phase 2 & 3 Complete)

All developer operations are now executable via Docker — **no host Python, Node.js, or CLI
tools required**.

#### `docker-compose.tools.yml` — Ad-hoc Developer CLI (profile: `tools`)

| Tool container | Purpose |
|----------------|---------|
| `ragas-pipeline` | Run full RAGAS testset generation |
| `run-pipeline` | Execute the main evaluation pipeline |
| `evaluate-testset` | Score an existing testset against metrics |
| `rag-cli` | Interactive RAG query CLI |
| `kg-tune` | Knowledge graph hyperparameter tuning |
| `perf-baseline` | Benchmark performance baseline recording |
| `validate-events` | Validate event schema registry |
| `validate-telemetry` | Validate telemetry contracts |
| `validate-tasks` | Validate task specification files |
| `validate-compose` | Lint all compose YAML files |
| `gen-openapi` | Generate OpenAPI specs from service code |
| `gen-supplychain` | Generate software supply-chain artefacts |
| `task-timeline` | Render task/milestone timeline |
| `gen-dashboard` | Regenerate dashboard static assets |
| `check-bundle` | Audit portal JS/CSS bundle sizes |

All tool containers reuse the existing `rag-eval:dev` image via bind-mount — zero additional
Dockerfiles required.

#### `docker-compose.init.yml` — One-Shot Init/Migration (profile: `init`)

| Init container | Purpose |
|----------------|---------|
| `db-migrate` | Reviewer state SQLite/Postgres schema migration |
| `model-preload` | Download sentence-transformer models to shared cache |
| `hash-schemas` | Recompute SHA-256 hashes in `events/schema_registry.json` |

#### `docker-compose.test.yml` — Isolated Clean-Room CI Environment

- `Dockerfile.test`: two-stage (deps/test) image on `python:3.10-slim-bullseye`.
- Corporate Squid proxy forwarded through `ARG HTTP_PROXY / HTTPS_PROXY`.
- CPU-only PyTorch wheel index (avoids pulling ~2 GB CUDA bundle).
- tiktoken BPE encoding files pre-warmed (`o200k_base` + `cl100k_base`).
- `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAGAS=0.2.15` for editable RAGAS install.
- Matches the environment of remote GitHub Actions runners exactly.

### 733-Test Parallel Suite

| Scope | Tests |
|-------|-------|
| `eval-pipeline/tests/` | 369 |
| `services/tests/` | 364 |
| **Total** | **733** |

- Parallel execution via `pytest-xdist` with worker isolation.
- Deterministic offline fallbacks for all LLM-dependent tests.
- CI-safe lazy import guards; `collect_ignore_glob` roots prevent accidental collection.
- Extraction failure CI gate: records and counts extraction errors as structured artefacts.

---

## 🧹 Chore & Refactoring

### Codebase Cleanup — ~200 Legacy Files Deleted (`35b101100`)

| Category | Files removed |
|----------|--------------|
| Session-progress `.md` files (`NEXT_STEPS_PRIORITY_Vx`, `*_COMPLETE.md`, …) | 99 |
| `eval-pipeline` one-off scripts (`debug_*.py`, `fix_*.py`, `apply_*.py`) | 42 |
| Root-level orphaned `test_*.py` scripts (superseded by `eval-pipeline/tests/`) | 27 |
| Runtime artefacts (`rag_evaluation_results_*.json`, `*.xlsx`, `*.log`) | 20 |
| Exploratory PNG charts committed by mistake | 15 |
| Legacy `eval-pipeline` Dockerfiles (`Dockerfile.optimized`, `.proxy-test`, …) | 5 |
| Detached sub-projects (`rr-data/`, `rr-eval/`, `rr-rag/`, `rr-tokenizer/`, `rr-agent-webui/`) | 5 |
| BFG rewrite artefacts (`.git.bfg-report/`, `bfg.jar`) | 1 |
| Stale portal `node_modules` snapshot | 1 |

The surviving codebase is fully intentional: `eval-pipeline/src/`, `eval-pipeline/tests/`,
`insights-portal/src/`, `services/`, `ragas/`, `config/`, `scripts/`, `docs/`, `deploy/`,
`helm/`.

### Additional Hardening

- **Path-traversal guard** in `evaluation_data_formatter` (`ffea16c`): input paths are now
  validated against the expected output directory root before file writes.
- **`ConfigManager.from_defaults()` factory classmethod** (`b8dda41`): eliminates fragile
  positional argument construction in downstream consumers.
- **Duplicate import and bare `except` removal** (`c54c0ae`, `5f67594`): code quality pass
  across the evaluation module tree.
- **`mypy` configuration** (`ffe380b`): focused `mypy.ini` with per-module overrides and
  incremental mode enabled.
- **`DEVELOPER_GUIDE.md`** (`2b4df69`, updated `62d4cbd`): comprehensive developer reference
  covering automated testing, UI testing, deployment, the GCR glossary, and copy-pasteable
  `docker compose run` examples for every tool and init container.

---

## 🔒 Security

- Path-traversal hardening in evaluation data formatter (OWASP A01/A03).
- API key validation with actionable 503 responses — no silent failures.
- Corporate proxy ARGs correctly scoped to builder stages to prevent secret leakage in final
  image layers.
- MinIO credentials and API keys exclusively via `env_file` — not baked into images.

---

## Upgrade Notes

This is the **initial production release** merging from `feat/graph-context-relevance`. There
is no prior `main` production state to migrate from. For first-time setup:

```bash
# 1. Copy and configure environment
cp config/pipeline_config.template.yaml config/pipeline_config.yaml

# 2. Run one-shot initialisation
docker compose -f docker-compose.init.yml --profile init up

# 3. Start all services
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml up -d

# 4. Run the full test suite
docker compose -f docker-compose.test.yml run --rm test

# 5. (Optional) Run any CLI tool, e.g. RAGAS pipeline
docker compose -f docker-compose.tools.yml --profile tools run --rm ragas-pipeline
```

---

## Acknowledgements

All 226 commits on this branch were authored by **Jason YY Lin** (@iiyyll01lin), with review
and architectural guidance from the Vibrant Labs AI team.

---

*Document generated: 2026-03-27*
