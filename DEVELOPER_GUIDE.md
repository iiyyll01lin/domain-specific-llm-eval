# Developer Guide — Domain-Specific LLM Evaluation System

> **Audience:** Engineers onboarding to this repository.  
> **Branch:** `feat/graph-context-relevance` → merges to `main`.

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Prerequisites](#2-prerequisites)
3. [Automated Testing](#3-automated-testing)
   - 3.1 [Full Unit/Integration Suite (733+ tests)](#31-full-unitintegration-suite-733-tests)
   - 3.2 [Golden-Path E2E Smoke Test](#32-golden-path-e2e-smoke-test)
   - 3.3 [Offline Tiktoken Pre-warming (Critical CI Trick)](#33-offline-tiktoken-pre-warming-critical-ci-trick)
4. [Manual UI Testing](#4-manual-ui-testing)
   - 4.1 [Starting the Insights Portal](#41-starting-the-insights-portal)
   - 4.2 [Loading JSON Evaluation Results](#42-loading-json-evaluation-results)
   - 4.3 [Loading CSV Reports](#43-loading-csv-reports)
5. [Service Deployment](#5-service-deployment)
   - 5.1 [Development Stack](#51-development-stack)
   - 5.2 [FastAPI Microservices](#52-fastapi-microservices)
   - 5.3 [GraphStore Backends](#53-graphstore-backends)
6. [GCR Metric Glossary](#6-gcr-metric-glossary)
7. [Project Structure Quick-Reference](#7-project-structure-quick-reference)
8. [Dockerized CLI Tools & Tasks](#8-dockerized-cli-tools--tasks)
   - 8.1 [Architecture: Three-File Compose Strategy](#81-architecture-three-file-compose-strategy)
   - 8.2 [Init Containers (docker-compose.init.yml)](#82-init-containers-docker-composeinityml)
   - 8.3 [Ad-hoc CLI Tools (docker-compose.tools.yml)](#83-ad-hoc-cli-tools-docker-composetoolsyml)
   - 8.4 [Webhook Daemon (always-on)](#84-webhook-daemon-always-on)
   - 8.5 [Makefile Shortcuts](#85-makefile-shortcuts)
9. [Auto-Insights: AI-Powered Executive Summary](#9-auto-insights-ai-powered-executive-summary)
   - 9.1 [Overview](#91-overview)
   - 9.2 [Enabling the Feature](#92-enabling-the-feature)
   - 9.3 [Model Selection & Custom Endpoints](#93-model-selection--custom-endpoints)
   - 9.4 [API Reference](#94-api-reference)
   - 9.5 [Architecture](#95-architecture)
10. [MLOps Drift Alerting (AIOps)](#10-mlops-drift-alerting-aiops)
    - 10.1 [Overview](#101-overview)
    - 10.2 [Architecture](#102-architecture)
    - 10.3 [Detection Algorithm](#103-detection-algorithm)
    - 10.4 [Environment Variables](#104-environment-variables)
    - 10.5 [REST Endpoint](#105-rest-endpoint)
    - 10.6 [Frontend Drift Monitor Banner](#106-frontend-drift-monitor-banner)
    - 10.7 [Slack Alert Format](#107-slack-alert-format)
    - 10.8 [Tuning the Detector](#108-tuning-the-detector)

---

## 1. Repository Overview

This system evaluates retrieval-augmented generation (RAG) pipelines against a
**Knowledge Graph (KG)** built from domain-specific documents (SMT/manufacturing CSV data).

```
CSV Documents
    │
    ▼
Knowledge Graph (SQLiteGraphStore)
    │
    ▼
GraphContextRelevanceEvaluator
    │  ─ Sₑ · Entity Overlap
    │  ─ Sc · Structural Connectivity
    │  └─ Ph · Hub Noise Penalty
    │
    ├──▶ JSON/CSV/XLSX Artifacts
    └──▶ insights-portal  (React dashboard)
```

Key source directories:

| Path | Purpose |
|------|---------|
| `eval-pipeline/src/evaluation/graph_context_relevance.py` | Core GCR evaluator |
| `eval-pipeline/src/utils/graph_store.py` | SQLite / Neo4j GraphStore |
| `eval-pipeline/src/evaluation/evaluation_dispatcher.py` | Unified metric routing |
| `eval-pipeline/tests/` | 733+ unit & integration tests |
| `eval-pipeline/scripts/golden_path_runner.py` | E2E smoke runner |
| `insights-portal/src/` | React + Vite dashboard |
| `docker-compose.test.yml` | Hermetic CI environment |
| `docker-compose.services.yml` | Production service stack |

---

## 2. Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Docker | 24.x | With BuildKit enabled |
| Docker Compose V2 | 2.20 | `docker compose` (not `docker-compose`) |
| Node.js | 18.x LTS | Only for local `npm run dev` (not needed for Docker) |
| Python | 3.10 | Only needed for local (non-Docker) runs |

Copy and populate the environment file:

```bash
cp .env.example .env
# Edit .env: set LLM endpoint, API keys if needed
```

---

## 3. Automated Testing

### 3.1 Full Unit/Integration Suite (733+ tests)

The entire test suite runs inside a clean-room Docker container — no Python
installed on the host, no internet access at runtime.

**Build the test image** (one-time, ~5 min):

```bash
docker compose -f docker-compose.test.yml build test
```

**Run the full suite:**

```bash
docker compose -f docker-compose.test.yml run --rm test
```

This internally executes:

```
python -m pytest eval-pipeline/tests/ -n auto --timeout=120 -q
```

Expected output: `732 passed, 1 skipped` in approximately 230 seconds.

**Run a specific test file:**

```bash
docker compose -f docker-compose.test.yml run --rm test \
  python -m pytest eval-pipeline/tests/test_graph_context_relevance.py -v
```

**Makefile shortcut** (from repo root):

```bash
# There is no single `make test` target yet — use docker compose directly.
# Alternatively, via eval-pipeline:
cd eval-pipeline && make build && docker compose -f docker-compose.test.yml run --rm test
```

---

### 3.2 Golden-Path E2E Smoke Test

The E2E smoke test exercises the **complete pipeline end-to-end** — from CSV
ingestion through graph construction, GCR evaluation, and artifact export — with
zero external dependencies.

```bash
docker compose -f docker-compose.test.yml run --rm e2e-smoke-test
```

What it validates:

1. Loads `eval-pipeline/tests/fixtures/golden_corpus.csv` (5 SMT documents)
2. Creates a clean `SQLiteGraphStore` (`golden_path.db`)
3. Ingests 5 graph nodes + 4 chain-topology edges (similarity scores 0.65–0.80)
4. Dispatches a `GraphSpec` through `EvaluationDispatcher`
5. Asserts all three GCR sub-scores are present and the composite score > 0
6. Writes CSV and XLSX artifacts via `PipelineFileSaver`

The generated artifacts persist in the named Docker volume `e2e-outputs`.
To copy them out for manual inspection:

```bash
# Identify the volume mount
docker volume inspect domain-specific-llm-eval_e2e-outputs

# Or use a helper container
docker run --rm -v domain-specific-llm-eval_e2e-outputs:/data alpine ls /data
```

The 6 pytest assertions are in `eval-pipeline/tests/e2e_golden_path.py`.

---

### 3.3 Offline Tiktoken Pre-warming (Critical CI Trick)

**The problem:** tiktoken downloads BPE encoding files (~1.7 MB) from an Azure
CDN on first import. In a CI environment without internet access this causes a
**137-second hang** followed by a timeout error.

**The solution baked into `Dockerfile.test`:**

```dockerfile
# During image BUILD (network available):
RUN python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base'); \
    tiktoken.get_encoding('cl100k_base')" \
    && cp -r /root/.cache/data-gym-cache /tmp/data-gym-cache
```

At runtime the cache is already on-disk; tiktoken finds it via the
`TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache` environment variable set in
`docker-compose.test.yml`.

**Offline environment variables** (all set in `docker-compose.test.yml`):

```yaml
HF_HUB_OFFLINE: "1"
TRANSFORMERS_OFFLINE: "1"
TIKTOKEN_CACHE_DIR: /tmp/data-gym-cache
MPLBACKEND: Agg        # prevents matplotlib from opening display windows
```

> **Rule of thumb:** If you add a dependency that downloads data on first
> import (spaCy models, NLTK corpora, etc.), add the same pre-warm step to
> `Dockerfile.test` under the `deps` build stage.

---

## 4. Manual UI Testing

### 4.1 Starting the Insights Portal

#### Option A — Full-Stack Docker (recommended)

The `insights-portal` is now a first-class service in the production compose
stack. Build and start the entire frontend + backend with a single command:

```bash
docker compose -f docker-compose.services.yml up -d --build
```

| Service | URL |
|---|---|
| Insights Portal (Nginx) | `http://localhost:5173` |
| Ingestion API | `http://localhost:8001` |
| Eval API | `http://localhost:8004` |
| Reporting API | `http://localhost:8005` |
| MinIO console | `http://localhost:9001` |

To rebuild only the frontend after UI changes:

```bash
docker compose -f docker-compose.services.yml build insights-portal
docker compose -f docker-compose.services.yml up -d insights-portal
```

#### Option B — Local dev server (hot-reload)

Use this during active frontend development to get instant HMR feedback:

```bash
cd insights-portal
npm install          # first time only
npm run dev          # starts Vite dev server on port 5173
```

The dev server binds to all interfaces (`host: true` in `vite.config.ts`).

For a local production preview build:

```bash
npm run build        # outputs to dist/
npm run preview      # serves dist/ on port 4173
```

---

### 4.2 Loading JSON Evaluation Results

The portal's primary input is a JSON evaluation result file produced by the pipeline.

**Source paths** (on the host after running the E2E test or full pipeline):

```
my_e2e_reports/evaluation_result.json     # E2E smoke test output
outputs/<run_timestamp>/evaluations/*.json # Full pipeline runs
```

**Steps in the UI:**

1. Open the portal (`http://localhost:5173`).
2. Click **"Load Results"** in the top navigation bar.
3. Select a `evaluation_result.json` or `rag_evaluation_results_*.json` file.
4. The dashboard populates with:
   - GCR composite score gauge
   - Per-metric breakdown (Sₑ, Sᶜ, Pₕ)
   - Retrieved node graph topology visualisation
   - Question/answer pairs with highlighted context

**Expected JSON schema** (top-level keys the portal requires):

```json
{
  "scores": {
    "gcr": 0.72,
    "se":  0.85,
    "sc":  0.67,
    "ph":  0.12
  },
  "backend": "GraphContextRelevanceEvaluator",
  "question": "...",
  "expected_answer": "...",
  "contexts": ["...", "..."]
}
```

---

### 4.3 Loading CSV Reports

The portal also accepts CSV summary reports (one row per evaluation run).

```
outputs/<run>/evaluations/*.csv           # from PipelineFileSaver
eval-pipeline/tests/fixtures/golden_corpus.csv  # reference corpus
```

**Steps:**

1. Click **"Import CSV"** on the dashboard sidebar.
2. Select the CSV file.
3. The portal renders a comparison table and trend chart if multiple rows are present.

**Tip:** The XLSX version (generated alongside the CSV) contains the same data
with multi-sheet formatting and is useful for sharing with non-technical stakeholders.

---

## 5. Service Deployment

### 5.1 Development Stack

Bring up all microservices with hot-reload:

```bash
make dev
# equivalent to:
docker compose -f docker-compose.services.yml \
               -f docker-compose.dev.override.yml up -d --build
```

Tear down:

```bash
docker compose -f docker-compose.services.yml down
```

---

### 5.2 FastAPI Microservices

The production service stack (`docker-compose.services.yml`) runs eight containers:

| Container | Role | Default Port |
|-----------|------|-------------|
| `rag-eval-ingestion` | Document ingest & chunking | 8001 |
| `rag-eval-processing` | Embedding + KG node creation | 8002 |
| `rag-eval-testset` | RAGAS testset generation | 8003 |
| `rag-eval-eval` | GCR & supplementary metric evaluation | 8004 |
| `rag-eval-reporting` | XLSX/CSV/HTML report generation | 8005 |
| `rag-eval-adapter` | External LLM proxy adapter | 8006 |
| `rag-eval-kg` | Knowledge Graph query API | 8007 |
| `rag-eval-webhook` | Webhook daemon + APScheduler drift monitor | 8008 |
| `rag-eval-insights-portal` | React evaluation dashboard (Nginx) | 5173 |
| `rag-eval-minio` | S3-compatible artifact store | 9000/9001 |

**Health check all services:**

```bash
for port in 8001 8002 8003 8004 8005 8006 8007; do
  echo -n "Port $port: "
  curl -sf http://localhost:$port/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "unreachable"
done
```

**Minimum required services** for just running GCR evaluation against an existing graph:

```bash
docker compose -f docker-compose.services.yml up -d rag-eval-kg rag-eval-eval
```

---

### 5.3 GraphStore Backends

The `GraphStore` is a Python `Protocol`; it can be swapped transparently.

**SQLite (default — no external dependencies):**

```python
from src.utils.graph_store import SQLiteGraphStore

store = SQLiteGraphStore("outputs/my_pipeline/graph.db")
# The DB file is created on first write. Content-hash addressing
# ensures idempotent node insertion.
```

**Neo4j (for production-scale graphs):**

```python
from src.utils.graph_store import Neo4jGraphStore

store = Neo4jGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
)
```

Configure which backend to use in `config/pipeline_config.yaml`:

```yaml
graph_store:
  backend: sqlite            # or "neo4j"
  sqlite_path: outputs/graph.db
  neo4j_uri: bolt://localhost:7687
```

**Connecting `GraphContextRelevanceEvaluator` to the store:**

```python
from src.evaluation.graph_context_relevance import GraphContextRelevanceEvaluator
from src.evaluation.evaluation_dispatcher import EvaluationDispatcher, GraphSpec

store = SQLiteGraphStore("outputs/graph.db")
evaluator = GraphContextRelevanceEvaluator(graph_store=store)
dispatcher = EvaluationDispatcher(graph_evaluator=evaluator)

result = dispatcher.dispatch(GraphSpec(
    question="What surface defects are detected on steel plates?",
    expected_answer="Scratches and pits are detected.",
    retrieved_node_ids=["node-1", "node-2", "node-5"],
))
print(result.scores)  # {"gcr": 0.71, "se": 0.84, "sc": 0.70, "ph": 0.10}
```

---

## 6. GCR Metric Glossary

The **Graph Context Relevance (GCR)** composite score measures how topologically
well-formed and semantically relevant the *retrieved subgraph* is for a given
question/answer pair.

$$
\text{GCR} = \text{clip}(\alpha \cdot S_e + \beta \cdot S_c - \gamma \cdot P_h,\; 0.0,\; 1.0)
$$

Default weights: **α = 0.4, β = 0.4, γ = 0.2**

---

### Sₑ — Entity Overlap (Semantic Relevance)

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Weight (α) | 0.4 |
| Fully offline | ✅ |

**Definition:** [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index)
between the token set of the question + expected answer and the aggregated token
sets of the retrieved nodes (keyphrases + entity labels + content terms).

$$
S_e = \frac{|T_Q \cap T_R|}{|T_Q \cup T_R|}
$$

where $T_Q$ = tokens from question & answer, $T_R$ = union of all retrieved node tokens.

**Interpretation:**
- `Sₑ ≈ 1.0` → retrieved nodes share nearly all domain terms with the query.
- `Sₑ ≈ 0.0` → retrieved nodes are topically unrelated to the question.

**Supports Chinese:** The tokeniser uses `[a-z0-9_\u4e00-\u9fff]+` so CJK
characters are treated as individual tokens, which is correct for entity matching.

---

### Sᶜ — Structural Connectivity (Coherence)

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Weight (β) | 0.4 |
| Fully offline | ✅ |

**Definition:** Ratio of the **largest connected component** to the total number
of retrieved nodes, computed over the undirected projection of edges whose *both*
endpoints are in the retrieved set.

$$
S_c = \frac{|\text{LCC}(G_R)|}{|V_R|}
$$

where $G_R$ is the subgraph induced by the retrieved node IDs and $V_R$ is the set
of retrieved nodes.

**Interpretation:**
- `Sc = 1.0` → all retrieved nodes form a single connected component (coherent retrieval).
- `Sc < 0.5` → retrieved nodes are scattered, fragmented — the context is likely noisy.

**Graph topology insight:** In a chain-topology graph (node-1 → node-2 → … → node-N),
retrieving all N nodes gives `Sc = 1.0`. Retrieving only the endpoints (node-1 and
node-N) gives `Sc = 0.5` if the middle nodes are absent.

---

### Pₕ — Hub Noise Penalty (Anti-Hub)

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Weight (γ) | 0.2 |
| Fully offline | ✅ |
| Applied as | Penalty (subtracted) |

**Definition:** Fraction of retrieved nodes classified as *degree hubs* — nodes
whose total degree in the **full** graph exceeds μ + 2σ (two standard deviations
above the mean degree).

$$
P_h = \frac{|\{v \in V_R : \deg(v) > \mu_d + 2\sigma_d\}|}{|V_R|}
$$

where $\mu_d$ and $\sigma_d$ are the mean and standard deviation of node degrees
over the full graph.

**Interpretation:**
- `Ph = 0.0` → no hub nodes retrieved; all retrieved context is specific.
- `Ph = 1.0` → every retrieved node is a hub; the context is likely dominated by
  generic "index" nodes that appear in many relationships and dilute precision.

**Why hubs are bad:** In a manufacturing KG, a node like "quality inspection" may
link to hundreds of sub-processes. Retrieving it provides no discriminative signal
for a specific question about "steel plate scratch detection."

---

### GCR — Composite Score

$$
\text{GCR} = \text{clip}(\underbrace{0.4 \cdot S_e}_{\text{semantic}} + \underbrace{0.4 \cdot S_c}_{\text{structural}} - \underbrace{0.2 \cdot P_h}_{\text{noise}},\; 0.0,\; 1.0)
$$

**Score bands:**

| GCR | Interpretation |
|-----|---------------|
| 0.80 – 1.00 | Excellent retrieval: highly relevant, coherent, low hub noise |
| 0.60 – 0.79 | Good retrieval: minor coherence or hub-noise issues |
| 0.40 – 0.59 | Marginal: significant fragmentation or off-topic nodes |
| 0.00 – 0.39 | Poor: retrieval is incoherent or completely off-topic |

**Tuning the weights:** Override defaults in `config/pipeline_config.yaml`:

```yaml
graph_context_relevance:
  alpha: 0.4   # Se weight
  beta:  0.4   # Sc weight
  gamma: 0.2   # Ph penalty weight
```

---

## 7. Project Structure Quick-Reference

```
domain-specific-llm-eval/
├── Dockerfile                  # Main application image
├── Dockerfile.test             # Clean-room test image (offline + CPU-only)
├── docker-compose.services.yml # Production microservice stack
├── docker-compose.test.yml     # Hermetic CI test environment
├── docker-compose.dev.override.yml  # Dev hot-reload overrides
├── Makefile                    # build / dev / compose shortcuts
├── README.md                   # Project overview
├── DEVELOPER_GUIDE.md          # ← you are here
├── conftest.py                 # Root pytest fixtures / path setup
├── e2e_smoke.py                # Lightweight smoke test entrypoint
│
├── eval-pipeline/
│   ├── src/
│   │   ├── evaluation/
│   │   │   ├── graph_context_relevance.py   # GCR: Sₑ, Sᶜ, Pₕ
│   │   │   ├── evaluation_dispatcher.py     # Unified routing (TestsetSpec)
│   │   │   └── ...
│   │   └── utils/
│   │       ├── graph_store.py               # SQLite + Neo4j GraphStore
│   │       └── ...
│   ├── tests/
│   │   ├── e2e_golden_path.py               # 6 golden-path E2E tests
│   │   ├── fixtures/
│   │   │   ├── golden_corpus.csv            # 5 SMT domain documents
│   │   │   └── golden_path_config.yaml      # minimal config for E2E
│   │   └── test_graph_*.py                  # unit tests for GCR + GraphStore
│   ├── scripts/
│   │   └── golden_path_runner.py            # E2E CLI runner (exit-coded)
│   ├── config/
│   │   └── pipeline_config.yaml             # Central configuration
│   ├── run_pure_ragas_pipeline.py           # RAGAS testset generation
│   └── requirements.txt
│
├── insights-portal/
│   ├── src/
│   │   ├── components/                      # React UI components
│   │   ├── core/metrics/registry.ts         # Metric type registry
│   │   └── ...
│   ├── vite.config.ts                       # host:true, port 5173
│   └── package.json
│
├── ragas/                                   # Modified RAGAS fork
├── services/                                # FastAPI microservice sources
├── config/                                  # Shared pipeline config
└── docs/                                    # Architecture diagrams
```

---

---

## 8. Dockerized CLI Tools & Tasks

This project follows a **Zero Host Dependency** architecture — no `python3 …` or
`bash …` commands are ever needed on the developer's host machine.  Every script
is accessible via `docker compose run`.

Three Compose files cover the complete surface:

| File | Purpose | Profile |
|------|---------|--------|
| `docker-compose.services.yml` | Always-on production services (incl. `webhook`) | _(none — always up)_ |
| `docker-compose.init.yml` | One-shot init / migration containers | `init` |
| `docker-compose.tools.yml` | Ad-hoc developer CLI tools | `tools` |

---

### 8.1 Architecture: Three-File Compose Strategy

```
docker compose up -d
  └─ docker-compose.services.yml   ← 10 always-on services (+ webhook daemon)

docker compose … run --rm <service>   ← triggered manually, never auto-starts
  ├─ docker-compose.init.yml          (profiles: ["init"])
  └─ docker-compose.tools.yml         (profiles: ["tools"])
```

All tool and init containers share the **same `rag-eval:dev` image** as the
production services — no extra build steps or separate Dockerfiles required.
Source code is bind-mounted read-only so your local edits are immediately
reflected without rebuilding.

**Common flags:**

| Flag | Meaning |
|------|--------|
| `--rm` | Delete the container after it exits (keeps things clean) |
| `--` | Separator between Compose args and the script's own CLI args |
| `-f A -f B` | Merge two Compose files (B overrides A for duplicate keys) |

---

### 8.2 Init Containers (`docker-compose.init.yml`)

Init containers are **idempotent, one-shot** tasks that prepare the environment.
Run them before the first `docker compose up`, or after the indicated trigger.

#### `model-preload` — Download sentence-transformer model (first deploy)

Required for offline embedding-based relationship building.  Needs outbound
HTTPS; set `HTTP_PROXY` / `HTTPS_PROXY` in your shell if behind a firewall.

```bash
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.init.yml \
  run --rm model-preload
```

#### `db-migrate` — Reviewer-state DB migration (after schema changes)

Creates or migrates the SQLite reviewer-state database.  Always exits 0 when
the schema is healthy; exits 1 on failure.

```bash
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.init.yml \
  run --rm db-migrate
```

To migrate to **Postgres** instead of SQLite:

```bash
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.init.yml \
  run --rm db-migrate \
  python scripts/reviewer_state_migrate.py \
    --backend postgres \
    --state-store-dsn postgresql://user:pass@host:5432/rageval
```

#### `hash-schemas` — Recompute event schema hashes

Run after adding or modifying any file under `eval-pipeline/events/schemas/`.
Updates the SHA-256 entries in `eval-pipeline/events/schema_registry.json`.

```bash
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.init.yml \
  run --rm hash-schemas
```

---

### 8.3 Ad-hoc CLI Tools (`docker-compose.tools.yml`)

All tool services carry `profiles: ["tools"]`, so `docker compose up -d` never
starts them automatically.

#### Pipeline Runners

```bash
# Generate a full RAGAS testset from CSV (needs LLM endpoint in .env.compose)
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm ragas-pipeline -- \
    --config config/pipeline_config.yaml --max-docs 50

# Simplified KG + testset pipeline
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm run-pipeline -- --config config/pipeline_config.yaml

# Re-evaluate an existing testset against a live RAG endpoint
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm evaluate-testset -- \
    --testset outputs/<run>/testsets/<file>.json

# Testset manager CLI
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm rag-cli -- list

docker compose ... run --rm rag-cli -- \
  validate outputs/<run>/testsets/<file>.json
```

#### KG & Scoring Tools

```bash
# Grid-search KG similarity thresholds
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm kg-tune -- \
    --input outputs/<run>/testsets/knowledge_graphs/<kg>.json \
    --output outputs/threshold_report.json

# Benchmark ingestion + relationship latency
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm perf-baseline -- --docs 20
```

Results land at `benchmarks/baseline.json` on the host (bind-mounted).

#### Code / Schema Validators (offline, fast)

```bash
# Validate event JSON schemas vs registry hashes (exit code 0 = pass)
docker compose -f docker-compose.tools.yml run --rm validate-events

# Validate telemetry taxonomy structure
docker compose -f docker-compose.tools.yml run --rm validate-telemetry

# Validate tasks.md governance blocks + EN/ZH parity
docker compose -f docker-compose.tools.yml run --rm validate-tasks

# Sanity-check docker-compose.services.yml service list
# ⚠ Requires Docker socket access; Docker daemon must be running on the host.
docker compose \
  -f docker-compose.services.yml \
  -f docker-compose.tools.yml \
  run --rm validate-compose
```

#### Artifact Generators

```bash
# Generate OpenAPI spec stubs for each FastAPI service
docker compose -f docker-compose.tools.yml run --rm gen-openapi

# Build SBOM diff + SLSA provenance artifacts
docker compose -f docker-compose.tools.yml run --rm gen-supplychain -- \
  --sbom sbom-current.json \
  --baseline-sbom sbom-baseline.json \
  --diff outputs/sbom_diff.json \
  --provenance outputs/provenance.json

# Parse tasks.md → task_timeline.json + .csv
docker compose -f docker-compose.tools.yml run --rm task-timeline

# Render sprint dashboard.html (run after task-timeline)
docker compose -f docker-compose.tools.yml run --rm gen-dashboard

# Enforce 300 KB gzip budget on KG panel JS chunk
# ⚠ Requires npm run build output inside insights-portal/dist/
docker compose -f docker-compose.tools.yml run --rm check-bundle
```

---

### 8.4 Webhook Daemon (always-on)

The `webhook` service is part of the **production stack** and runs 24/7 alongside
the other microservices.  It is a FastAPI application that listens for CI push
events and triggers the RAGAS evaluation pipeline as a background task.

| Service | URL | Log file |
|---------|-----|----------|
| `rag-eval-webhook` | `http://localhost:8008` | `outputs/webhook_events.jsonl` |

**Start with the main stack:**

```bash
docker compose -f docker-compose.services.yml up -d
# Webhook daemon starts automatically along with all other services.
```

**Health check:**

```bash
curl http://localhost:8008/health
# Expected: {"status": "ok"}
```

**Trigger a pipeline run manually:**

```bash
curl -X POST http://localhost:8008/webhook \
  -H "Content-Type: application/json" \
  -d '{"event_type": "manual", "ref": "refs/heads/main", "docs": 10, "samples": 50}'
```

**Inspect the event log:**

```bash
tail -f outputs/webhook_events.jsonl | python3 -m json.tool
```

---

### 8.5 Makefile Shortcuts

Add these targets to your `Makefile` for convenience:

```makefile
# Init
model-preload:
	docker compose -f docker-compose.services.yml -f docker-compose.init.yml run --rm model-preload

db-migrate:
	docker compose -f docker-compose.services.yml -f docker-compose.init.yml run --rm db-migrate

hash-schemas:
	docker compose -f docker-compose.services.yml -f docker-compose.init.yml run --rm hash-schemas

# Validation (CI-friendly, fully offline)
validate-events:
	docker compose -f docker-compose.tools.yml run --rm validate-events

validate-telemetry:
	docker compose -f docker-compose.tools.yml run --rm validate-telemetry

validate-tasks:
	docker compose -f docker-compose.tools.yml run --rm validate-tasks

# Artifact generation
task-timeline:
	docker compose -f docker-compose.tools.yml run --rm task-timeline

gen-dashboard: task-timeline
	docker compose -f docker-compose.tools.yml run --rm gen-dashboard
```

---

*Generated: 2026-03-26 | Branch: `feat/graph-context-relevance`*

---

## 9. Auto-Insights: AI-Powered Executive Summary

### 9.1 Overview

The **Auto-Insights** feature sends aggregated evaluation KPIs to an LLM (OpenAI
GPT-4o-mini by default, configurable) and renders a human-readable **System Health
Report** at the top of the Executive Overview dashboard.

The report interprets every metric — including the domain-specific Graph Context
Relevance suite (Sₑ, Sᶜ, Pₕ) — and produces:

- A single-sentence **overall health verdict** (🟢 Healthy / 🟡 At Risk / 🔴 Critical)
- **Key Findings** — 3–5 bullet points citing specific metric values
- **Recommended Actions** — prioritised engineering steps ordered by impact

The feature is **opt-in**. If no API key is configured the UI displays a friendly
setup prompt instead of crashing.

---

### 9.2 Enabling the Feature

**1. Set your API key.**

Create or edit the `.env` file at the repository root (it is git-ignored):

```dotenv
# Required — OpenAI key (or any compatible provider key)
OPENAI_API_KEY=sk-...your-key-here...
```

Alternatively you can use the alias `INSIGHTS_API_KEY` if you want to keep your
primary `OPENAI_API_KEY` separate from the insights feature:

```dotenv
INSIGHTS_API_KEY=sk-...your-key-here...
```

**2. Rebuild the reporting service container.**

```bash
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml \
  up -d --build reporting
```

**3. Reload the dashboard.**

Load an evaluation run in the **Executive Overview** tab.  A `✨ Generate Executive
Summary` button will appear at the top of the page. Click it — results appear in
~5–20 seconds depending on model latency.

---

### 9.3 Model Selection & Custom Endpoints

The following env vars (set in `.env` or injected into the container) control LLM
behaviour:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Primary OpenAI (or compatible) API key |
| `INSIGHTS_API_KEY` | — | Alias; used if `OPENAI_API_KEY` is not set |
| `INSIGHTS_LLM_MODEL` | `gpt-4o-mini` | Model name passed to the API |
| `INSIGHTS_API_BASE_URL` | *(OpenAI default)* | Override base URL — use for Azure OpenAI, local `vllm`, or Ollama proxies |

**Example: Claude 3.5 Sonnet via a compatible proxy**

```dotenv
OPENAI_API_KEY=your-anthropic-key
INSIGHTS_API_BASE_URL=https://api.anthropic.com/v1
INSIGHTS_LLM_MODEL=claude-3-5-sonnet-20241022
```

**Example: Local Ollama**

```dotenv
INSIGHTS_API_KEY=ollama
INSIGHTS_API_BASE_URL=http://host.docker.internal:11434/v1
INSIGHTS_LLM_MODEL=qwen2.5:72b
```

> **Cost note:** `gpt-4o-mini` costs roughly $0.0001–$0.0003 per summary call
> (< 800 input tokens + < 600 output tokens). `gpt-4o` produces richer prose at
> ~10× the cost. For high-volume CI pipelines, `gpt-4o-mini` is recommended.

---

### 9.4 API Reference

**Endpoint:** `POST http://localhost:8005/api/v1/insights/generate`

**Request body (JSON):**

```jsonc
{
  "run_id": "run_20260327_143200",       // optional
  "kpis": {
    "gcr_score": 0.612,
    "entity_overlap": 0.38,
    "structural_connectivity": 0.71,
    "hub_noise_penalty": 0.41,
    "Faithfulness": 0.65,
    "AnswerRelevancy": 0.82
  },
  "verdict": "At Risk",                  // optional
  "failing_metrics": ["Faithfulness"],   // optional
  "thresholds": {                        // optional — enriches LLM context
    "Faithfulness": { "warning": 0.7, "critical": 0.5 }
  },
  "model": "gpt-4o"                      // optional — per-request model override
}
```

**Success response (200):**

```json
{
  "run_id": "run_20260327_143200",
  "summary": "🟡 **At Risk**: ...",
  "model_used": "gpt-4o-mini",
  "prompt_tokens": 512,
  "completion_tokens": 310
}
```

**Error responses:**

| HTTP | Condition |
|---|---|
| `503` | No API key configured; body contains setup instructions |
| `502` | Upstream LLM returned an error |
| `422` | Malformed request body |

---

### 9.5 Architecture

```
Insights Portal (React)
  └─ AiInsightCard component
       │  (click "Generate Executive Summary")
       ▼
  POST /api/v1/insights/generate
       │  (reporting service — port 8005)
       ▼
  _build_user_message()  ← formats KPIs + thresholds as structured Markdown
       │
       ▼
  OpenAI-compatible LLM
  (system prompt teaches metric semantics including Sₑ, Sᶜ, Pₕ direction)
       │
       ▼
  InsightsResponse { summary: Markdown string }
       │
       ▼
  AiInsightCard renders Markdown inline (no external deps)
  with animated gradient border while generating
```

**Key source files:**

| File | Purpose |
|---|---|
| `services/reporting/main.py` | FastAPI endpoint + LLM System Prompt |
| `insights-portal/src/components/AiInsightCard.tsx` | React card component |
| `insights-portal/src/app/lifecycle/api.ts` | `generateAiInsights()` fetch helper |
| `insights-portal/src/styles/theme.css` | Gradient border animation CSS |

---

## 10. MLOps Drift Alerting (AIOps)

### 10.1 Overview

Production RAG systems experience **Data Drift**: the distribution of real user
queries shifts over time.  When new query batches fall outside the Knowledge
Graph's domain, the topological GCR metrics degrade silently:

| Symptom | Metric | Expected behaviour |
|---------|--------|--------------------|
| Queries use vocabulary absent from the KG | Sₑ ↓ | Entity Overlap drops |
| Retrieved nodes become topologically isolated | Sᶜ ↓ | Structural Connectivity collapses |
| Generic hub nodes dominate retrieval | Pₕ ↑ | Hub Noise Penalty spikes |

The **Drift Alerting** feature monitors these trends automatically and notifies
administrators — both in the React dashboard and via Slack — before user-facing
quality degrades.

---

### 10.2 Architecture

```
outputs/run_*/kpis.json          ← per-run GCR metric averages
        │
        ▼
services/eval/drift/store.py     ← DriftStore (file scanner)
        │
        ▼
services/eval/drift/detector.py  ← DriftDetector (Welch Z-test)
        │
        ├──▶ drift/notifier.py   ← Slack webhook POST
        │
        └──▶ drift/scheduler.py  ← APScheduler BackgroundScheduler
                │
                ▼
eval-pipeline/webhook_daemon.py  ← lifespan: start/stop scheduler
                │                   GET /api/v1/drift-status
                ▼
insights-portal/src/components/DriftMonitorBanner.tsx
                                 ← polls /api/v1/drift-status every 5 min
```

Key source files:

| File | Purpose |
|------|---------|
| `services/eval/drift/store.py` | Scan `outputs/` and parse `kpis.json` averages |
| `services/eval/drift/detector.py` | Welch Z-score drift detection, severity roll-up |
| `services/eval/drift/notifier.py` | Slack incoming-webhook dispatcher |
| `services/eval/drift/scheduler.py` | APScheduler wiring + module-level result cache |
| `eval-pipeline/webhook_daemon.py` | FastAPI lifespan + `GET /api/v1/drift-status` |
| `insights-portal/src/components/DriftMonitorBanner.tsx` | React banner component |

---

### 10.3 Detection Algorithm

For each tracked metric $m \in \{S_e, S_c, P_h\}$, given:
- **Baseline window** $\mathcal{B}$ = the oldest $N$ runs (default $N = 100$)
- **Recent window** $\mathcal{R}$ = the latest $k$ runs (default $k = 50$)

$$\mu_B = \frac{1}{N}\sum_{i=1}^{N} m(b_i), \qquad \sigma_B = \sqrt{\frac{1}{N{-}1}\sum_{i=1}^{N}(m(b_i)-\mu_B)^2}$$

$$z_m = \frac{\bar{m}_R - \mu_B}{\sigma_B / \sqrt{k}}$$

**Directional flags** (threshold $\theta = 2.0$):

| Metric | Flag when |
|--------|-----------|
| $S_e$, $S_c$ (higher is better) | $z_m < -\theta$ |
| $P_h$ (lower is better) | $z_m > +\theta$ |

**Severity roll-up:**

| Flags | Status |
|-------|--------|
| 0 | `HEALTHY` |
| 1 | `WARNING` |
| ≥ 2 | `DRIFTING` |

When fewer than `min_baseline + 1` runs exist the status is `INSUFFICIENT_DATA`
— no alert is ever fired from this state.

---

### 10.4 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_WEBHOOK_URL` | _(empty — disabled)_ | Slack incoming-webhook URL. When set, an alert is POSTed whenever drift status is `WARNING` or `DRIFTING`. |
| `DRIFT_CHECK_INTERVAL_HOURS` | `6` | How often the background scheduler runs the drift check. Minimum value is 1 hour. |

Set these in the `.env` file at the repository root (git-ignored):

```dotenv
# Send drift alerts to the #rag-ops Slack channel
# Obtain the webhook URL from your Slack app settings: Apps > Incoming Webhooks
SLACK_WEBHOOK_URL=<your-slack-incoming-webhook-url>

# Check every 2 hours instead of 6
DRIFT_CHECK_INTERVAL_HOURS=2
```

Or inject them directly into the `webhook` container:

```yaml
# docker-compose.dev.override.yml
services:
  webhook:
    environment:
      SLACK_WEBHOOK_URL: "${SLACK_WEBHOOK_URL}"
      DRIFT_CHECK_INTERVAL_HOURS: "2"
```

---

### 10.5 REST Endpoint

**`GET http://localhost:8008/api/v1/drift-status`**

Returns the cached result of the most recent drift check.

```jsonc
// Example — DRIFTING
{
  "status": "DRIFTING",
  "checked_at": "2026-03-27T06:00:00+00:00",
  "baseline_window_size": 80,
  "recent_window_size": 50,
  "message": "Data drift detected — Structural Connectivity (Sᶜ): -31.2% (z=-3.41).",
  "metrics": {
    "structural_connectivity": {
      "metric": "structural_connectivity",
      "baseline_mean": 0.712,
      "recent_mean": 0.490,
      "baseline_std": 0.065,
      "z_score": -3.41,
      "delta_pct": -31.2,
      "flagged": true
    },
    "entity_overlap": {
      "metric": "entity_overlap",
      "baseline_mean": 0.381,
      "recent_mean": 0.370,
      "baseline_std": 0.048,
      "z_score": -1.62,
      "delta_pct": -2.9,
      "flagged": false
    }
  }
}

// Cold start (no check yet)
{ "status": "PENDING", "message": "Drift check not yet completed." }

// Module not installed
{ "status": "UNAVAILABLE", "message": "Drift monitoring module not installed." }
```

---

### 10.6 Frontend Drift Monitor Banner

The `DriftMonitorBanner` component renders immediately above the KPI cards in the
**Executive Overview** tab.

| Status | Appearance | Detail shown |
|--------|-----------|--------------|
| `HEALTHY` | Subtle green bar | Collapsed by default — click ▼ to expand |
| `WARNING` | Amber bar | Auto-expanded, shows per-metric delta table |
| `DRIFTING` | Red bar | Auto-expanded + **Action Required** CTA |
| `INSUFFICIENT_DATA` | Grey bar | Collapsed — shows reason text |
| `UNAVAILABLE` | Grey bar | Collapsed — webhook daemon unreachable |

The banner polls the webhook daemon every **5 minutes** using a browser
`fetch()` call.  The webhook URL defaults to `http://localhost:8008` and can be
overridden with the Vite env variable `VITE_WEBHOOK_BASE`.

---

### 10.7 Slack Alert Format

When status is `WARNING` or `DRIFTING` the notifier fires a Slack mrkdwn message:

```
🚨 *Data Drift Alert — `DRIFTING`*

• *Structural Connectivity (Sᶜ)*: recent `0.490` vs baseline `0.712` (-31.2%, z=-3.41)

⚠️ *Action Required:* Analyze failing queries and inject new domain documents into the Knowledge Graph.
_Checked at: 2026-03-27T06:00:00+00:00_
```

The webhook URL is never logged, stored in code, or committed to the repository.

---

### 10.8 Tuning the Detector

Override defaults via constructor arguments when calling `DriftDetector` directly:

```python
from services.eval.drift.detector import DriftDetector

detector = DriftDetector(
    baseline_n=50,        # use 50 runs for baseline (smaller corpus)
    recent_k=20,          # compare against last 20 runs
    z_threshold=2.5,      # tighten alert threshold (fewer false positives)
    min_baseline=10,      # need at least 11 runs before alerting
)
```

*Generated: 2026-03-27 | Branch: `feat/graph-context-relevance`*
