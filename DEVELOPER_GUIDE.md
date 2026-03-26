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
| Node.js | 18.x LTS | For the insights-portal only |
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

Expected output: `733 passed, 0 failed` in approximately 230 seconds.

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

```bash
cd insights-portal
npm install          # first time only
npm run dev          # starts Vite dev server
```

The dev server binds to **all interfaces** (`host: true` in `vite.config.ts`)
on port **5173**.

Access the portal at:
- `http://localhost:5173` (from the same machine)
- `http://<HOST_IP>:5173` (from a browser on another machine, if port 5173 is open)

For a production preview build:

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
   - Per-metric breakdown (Sₑ, Sc, Ph)
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

### Sc — Structural Connectivity (Coherence)

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

### Ph — Hub Noise Penalty (Anti-Hub)

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
│   │   │   ├── graph_context_relevance.py   # GCR: Sₑ, Sc, Ph
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

*Generated: 2026-03-26 | Branch: `feat/graph-context-relevance`*
