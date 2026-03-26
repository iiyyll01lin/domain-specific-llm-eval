# Infrastructure Guide — Service Topology, Storage, and Developer Tooling

> **Project:** Domain-Specific RAG Evaluation & MLOps Platform  
> **Version:** 1.1.0  
> **Last updated:** 2026-03-27

---

## Table of Contents

1. [Service Topology Overview](#service-topology-overview)
2. [Port Map](#port-map)
3. [Docker Compose Files — Purpose and Usage](#docker-compose-files--purpose-and-usage)
4. [Starting the Stack](#starting-the-stack)
5. [Content-Hash-Addressed SQLite GraphStore](#content-hash-addressed-sqlite-graphstore)
6. [docker-compose.tools.yml — Developer CLI Tooling](#docker-compose-toolsyml--developer-cli-tooling)
7. [docker-compose.init.yml — One-Shot Initialisation](#docker-compose-inityml--one-shot-initialisation)
8. [docker-compose.test.yml — Clean-Room CI Environment](#docker-compose-testyml--clean-room-ci-environment)
9. [Volume Strategy](#volume-strategy)
10. [Corporate Proxy Support](#corporate-proxy-support)
11. [Security Architecture](#security-architecture)

---

## Service Topology Overview

The platform is composed of **11 services** orchestrated via Docker Compose, all communicating
over the `rag-eval-net` bridge network. Every Python service shares a single `rag-eval:dev` image
built from the root `Dockerfile`, with the active service selected via the `SERVICE` environment
variable.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         rag-eval-net (bridge)                            │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ ingestion  │  │ processing │  │  testset   │  │       eval         │ │
│  │  :8001     │  │  :8002     │  │  :8003     │  │      :8004         │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ reporting  │  │  adapter   │  │     kg     │  │      webhook       │ │
│  │  :8005     │  │  :8006     │  │  :8007     │  │      :8008         │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                                          │
│  ┌────────────────────┐  ┌──────────────────────────────────────────┐   │
│  │  insights-portal   │  │     minio (+ minio-init)                 │   │
│  │      :5173         │  │     :9000 (API) / :9001 (Console)        │   │
│  └────────────────────┘  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Port Map

| Service | Host Port | Container Port | Role |
|---------|-----------|----------------|------|
| `ingestion` | **8001** | 8000 | Document submission with SHA-256 checksum deduplication |
| `processing` | **8002** | 8000 | Async document processing workers |
| `testset` | **8003** | 8000 | RAGAS testset generation (CSV → Knowledge Graph → testset) |
| `eval` | **8004** | 8000 | Metric evaluation — GCR, RAGAS, hybrid, contextual keyword |
| `reporting` | **8005** | 8000 | Aggregation, artefact storage, AI Insights (`POST /api/v1/insights/generate`) |
| `adapter` | **8006** | 8000 | External LLM provider adapter (OpenAI / Azure / vLLM / Ollama) |
| `kg` | **8007** | 8000 | Knowledge Graph builder & query API |
| `webhook` | **8008** | 8000 | Git webhook daemon (`POST /webhook`, `GET /health`, `GET /api/v1/drift-status`) |
| `insights-portal` | **5173** | 80 | React SPA served by Nginx |
| `minio` | **9000** | 9000 | S3-compatible artefact object store API |
| `minio` | **9001** | 9001 | MinIO web console |
| `minio-init` | — | — | One-shot bucket provisioning (no persistent port) |

---

## Docker Compose Files — Purpose and Usage

| File | Purpose | When to use |
|------|---------|-------------|
| `docker-compose.services.yml` | **Production service definitions** — all 11 containers | Always: base of every compose invocation |
| `docker-compose.dev.override.yml` | Bind-mounts repo root + enables `--reload` for hot code changes | Local development |
| `docker-compose.tools.yml` | Ad-hoc CLI tools (profile: `tools`) — never auto-started | Running one-off pipeline commands |
| `docker-compose.init.yml` | One-shot initialisation (profile: `init`) — DB migrations, model preload | First-time setup or schema upgrades |
| `docker-compose.test.yml` | Hermetic parallel test runner + E2E smoke test service | CI/CD and local test runs |

---

## Starting the Stack

### First-time setup

```bash
# 1. Copy and configure environment
cp config/pipeline_config.template.yaml config/pipeline_config.yaml
# Edit pipeline_config.yaml — set LLM endpoint, credentials, etc.

# 2. Run one-shot initialisation (DB migrations + model preload)
docker compose -f docker-compose.init.yml --profile init up

# 3. Start all services with hot-reload (development mode)
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml up -d --build
```

### Health check all services

```bash
for port in 8001 8002 8003 8004 8005 8006 8007 8008; do
  echo -n "Port $port: "; curl -sf http://localhost:$port/health && echo OK || echo FAIL
done
```

### View live logs

```bash
docker compose -f docker-compose.services.yml logs -f eval reporting webhook
```

### Stop all services

```bash
docker compose -f docker-compose.services.yml down
```

---

## Content-Hash-Addressed SQLite GraphStore

### Design Goals

The `SQLiteGraphStore` (`eval-pipeline/src/utils/graph_store.py`) is the persistent backend for
the Knowledge Graph evaluation layer. Its design intentionally prioritises:

1. **Zero new dependencies** — uses Python's stdlib `sqlite3`.
2. **Content-hash addressing** — each node's primary key is derived from its content, not
   an auto-increment ID, ensuring identical chunks are never stored twice.
3. **Protocol-based interface** — callers depend on the `GraphStore` Protocol; swapping to
   `Neo4jGraphStore` or any future backend requires no changes at call sites.

### Content Hashing

```python
def hash_content(text: str) -> str:
    """Return the first 32 hex chars of SHA-256(text.strip())."""
    normalised = text.strip().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()[:32]
```

- **32 hex chars = 128 bits of address space.** Birthday-paradox collision probability at $10^6$ chunks ≈ $1.5 \times 10^{-27}$.
- **Whitespace normalisation** before hashing: chunks differing only in leading/trailing whitespace hash identically (deduplication feature).
- Used as `TEXT PRIMARY KEY` in the `nodes` table — SQLite B-tree index provides $O(\log N)$ lookup.

### Schema

```sql
CREATE TABLE IF NOT EXISTS nodes (
    node_hash  TEXT PRIMARY KEY,
    node_type  TEXT NOT NULL,
    properties TEXT NOT NULL,    -- JSON-serialised dict
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_hash   TEXT NOT NULL REFERENCES nodes(node_hash) ON DELETE CASCADE,
    tgt_hash   TEXT NOT NULL REFERENCES nodes(node_hash) ON DELETE CASCADE,
    rel_type   TEXT NOT NULL,
    properties TEXT NOT NULL,    -- JSON-serialised metadata (scores, entities, …)
    created_at TEXT NOT NULL,
    UNIQUE(src_hash, tgt_hash, rel_type)    -- idempotent insertions
);

CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(src_hash);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relationships(tgt_hash);
```

### Key Behaviours

| Operation | Behaviour |
|-----------|-----------|
| `upsert_node(hash, type, props)` | INSERT OR UPDATE; auto-generates `_node_uuid` if absent (RAGAS compatibility) |
| `add_relationship(src, tgt, type, props)` | `INSERT OR IGNORE` — idempotent; duplicate `(src, tgt, type)` triples are silently dropped |
| `node_exists(hash)` | Single-row `LIMIT 1` lookup — $O(\log N)$ |
| `filter_new_hashes(candidates)` | Batch-check which hashes are not yet stored; used for incremental ingestion |
| `prune_stale(max_age_days)` | Deletes nodes (and cascades to relationships) not updated within the window |
| `get_all_nodes()` | Full table scan — result loaded into NetworkX graph once, then cached |

### Transaction Model

All writes are wrapped in an explicit context-manager connection with:
- `PRAGMA foreign_keys = ON` — enforces referential integrity on cascade deletes
- `conn.commit()` on success, `conn.rollback()` on any exception
- `conn.close()` in the `finally` block — no connection pooling needed for the batch workload

---

## docker-compose.tools.yml — Developer CLI Tooling

All tool containers use the `profiles: ["tools"]` flag — they **never start automatically** with
`docker compose up`. Invoke them individually with `run --rm`.

### General invocation pattern

```bash
docker compose \
    -f docker-compose.services.yml \
    -f docker-compose.tools.yml \
    run --rm <service-name> [-- <extra-args>]
```

### Pipeline Runners

| Container | Command | Description |
|-----------|---------|-------------|
| `ragas-pipeline` | `run --rm ragas-pipeline` | Full RAGAS testset generation: CSV → KG → testset artifacts |
| `run-pipeline` | `run --rm run-pipeline` | Simplified KG + testset pipeline runner |
| `evaluate-testset` | `run --rm evaluate-testset` | Evaluate an existing testset against a RAG endpoint |
| `rag-cli` | `run --rm rag-cli list` | Testset manager utility (list / validate / evaluate) |

### KG & Scoring Tools

| Container | Command | Description |
|-----------|---------|-------------|
| `kg-tune` | `run --rm kg-tune` | Grid-search KG similarity thresholds; writes `benchmarks/threshold_report.json` |
| `perf-baseline` | `run --rm perf-baseline` | Benchmark extraction + relationship latency; writes `benchmarks/baseline.json` |

### Code & Schema Validators (fully offline)

```bash
# Validate all event JSON schemas against the registry
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm validate-events

# Validate telemetry taxonomy contracts
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm validate-telemetry

# Validate tasks.md governance blocks
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm validate-tasks

# Lint docker-compose.services.yml service list
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm validate-compose
```

### Artifact Generators

```bash
# Generate OpenAPI specs for all FastAPI services
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm gen-openapi

# Build SBOM diff + SLSA provenance JSON artifacts
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm gen-supplychain

# Parse tasks.md → task_timeline.json + task_timeline.csv
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm task-timeline

# Regenerate sprint dashboard HTML from timeline JSON
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm gen-dashboard

# Enforce gzip size budget on the KG panel JS chunk
# (requires npm run build output in insights-portal/dist/)
docker compose -f docker-compose.services.yml -f docker-compose.tools.yml \
    run --rm check-bundle
```

---

## docker-compose.init.yml — One-Shot Initialisation

Use the `init` profile for first-time setup and schema migrations. These containers exit after
completion — they are not persistent services.

```bash
# Run all init containers
docker compose -f docker-compose.init.yml --profile init up

# Run a specific init step
docker compose -f docker-compose.init.yml --profile init run --rm db-migrate
docker compose -f docker-compose.init.yml --profile init run --rm model-preload
docker compose -f docker-compose.init.yml --profile init run --rm hash-schemas
```

| Container | Purpose |
|-----------|---------|
| `db-migrate` | Reviewer state SQLite / Postgres schema migration |
| `model-preload` | Download `sentence-transformers` models to the shared `models-cache` volume |
| `hash-schemas` | Recompute SHA-256 hashes in `events/schema_registry.json` |

---

## docker-compose.test.yml — Clean-Room CI Environment

The test runner is fully hermetic — no host Python, models, or network access required.

### Running the full test suite

```bash
# Build (once, or after requirements change)
docker compose -f docker-compose.test.yml build

# Run all 733 tests in parallel
docker compose -f docker-compose.test.yml run --rm test

# Run a single test file interactively
docker compose -f docker-compose.test.yml run --rm test \
    pytest eval-pipeline/tests/test_graph_context_relevance.py -v -s

# Run with coverage report
docker compose -f docker-compose.test.yml run --rm test \
    pytest -n auto --cov=eval-pipeline/src --cov-report=term-missing
```

### Offline isolation gates

The test container enforces strict offline mode via environment variables:

| Variable | Value | Effect |
|----------|-------|--------|
| `HF_HUB_OFFLINE` | `1` | Blocks all HuggingFace Hub download attempts |
| `TRANSFORMERS_OFFLINE` | `1` | Blocks all `transformers` model downloads |
| `HF_DATASETS_OFFLINE` | `1` | Blocks all dataset downloads |
| `TIKTOKEN_CACHE_ONLY` | `1` | Tiktoken reads only from pre-warmed BPE cache files |
| `TIKTOKEN_DISABLE_DOWNLOAD` | `1` | Prevents tiktoken from attempting network fetches |

### Test suite breakdown

| Scope | Count | Location |
|-------|-------|----------|
| Eval pipeline | 369 | `eval-pipeline/tests/` |
| Services | 364 | `services/tests/` |
| **Total** | **733** | Collected in 4.10s |

---

## Volume Strategy

| Volume | Type | Purpose |
|--------|------|---------|
| `minio-data` | Named | MinIO object storage (artefacts, testsets, reports) |
| `models-cache` | Named | Shared sentence-transformer model files across all services |
| `test-outputs` | Named | Writable scratch space for test artefacts (avoids host pollution) |
| `test-logs` | Named | Log files from test runs |
| `./.cache` | Bind | tiktoken BPE files, HuggingFace hub cache (offline use) |
| `./outputs` | Bind | Pipeline run outputs (`outputs/run_YYYYMMDD_HHMMSS_*/`) |
| `./eval-pipeline` | Bind (dev) | Hot-reload: source edits are live without image rebuild |

---

## Corporate Proxy Support

All `Dockerfile` `ARG` stages and compose `build.args` accept proxy configuration:

```bash
# Build with proxy
HTTP_PROXY=http://10.6.254.210:3128 \
HTTPS_PROXY=http://10.6.254.210:3128 \
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml \
    up -d --build
```

Proxy `ARG`s are scoped to the **builder stages only** (`RUN pip install`, `apt-get`) — they
are **not** baked into the final image layer's `ENV`. This prevents proxy credentials from
leaking into the production image and being exposed via `docker inspect`.

---

## Security Architecture

| Concern | Implementation |
|---------|---------------|
| **Path traversal** | `evaluation_data_formatter` validates all output paths against the expected root before writes (OWASP A01/A03) |
| **API key handling** | All secrets via `env_file: .env.compose` — never baked into images or committed |
| **MinIO credentials** | `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` set via env_file; defaults only for local dev |
| **CORS** | Explicitly configured on `reporting` service for cross-origin browser fetch |
| **Missing API key** | Returns actionable `503` with `"Set INSIGHTS_API_KEY"` message — no silent failures |
| **Proxy ARG scoping** | Build-time `HTTP_PROXY` / `HTTPS_PROXY` confined to builder stage; absent from final layer |
| **Reviewer auth** | JWT-based authentication on reviewer service endpoints |
