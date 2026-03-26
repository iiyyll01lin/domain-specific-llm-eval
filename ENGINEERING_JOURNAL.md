# Engineering Journal — Development History & Architectural Decisions

> **Project:** Domain-Specific RAG Evaluation & MLOps Platform  
> **Version:** 1.1.0  
> **Branch:** `feat/graph-context-relevance` → `main`  
> **Total commits:** 226 | **Files changed:** 724 | **Lines:** 139,971 insertions, 276 deletions  
> **Last updated:** 2026-03-27

---

## Table of Contents

1. [Project Genesis — From Prototype to Platform](#project-genesis--from-prototype-to-platform)
2. [Milestone 1: Zero-Host-Dependency Initiative](#milestone-1-zero-host-dependency-initiative)
3. [Milestone 2: The Legacy Purge — Deleting 200+ Files](#milestone-2-the-legacy-purge--deleting-200-files)
4. [Milestone 3: GCR Metric — The Core Innovation](#milestone-3-gcr-metric--the-core-innovation)
5. [Milestone 4: Tiktoken Offline Cache Fix](#milestone-4-tiktoken-offline-cache-fix)
6. [Milestone 5: Drift Detection System](#milestone-5-drift-detection-system)
7. [Milestone 6: Frontend — Insights Portal Overhaul](#milestone-6-frontend--insights-portal-overhaul)
8. [Milestone 7: AI Auto-Insights Implementation](#milestone-7-ai-auto-insights-implementation)
9. [Key Architectural Decisions (ADRs)](#key-architectural-decisions-adrs)
10. [Known Technical Debt & Planned Improvements](#known-technical-debt--planned-improvements)

---

## Project Genesis — From Prototype to Platform

The project began as an exploratory RAG evaluation script collection — a set of standalone
Python files that loaded documents, ran RAGAS evaluations, and dumped JSON results. By the
time the `feat/graph-context-relevance` branch was cut, the state of the codebase was:

- **~20 standalone scripts** in the root `eval-pipeline/` directory with overlapping
  functionality (`generate_dataset_configurable.py`, `generate_synthetic_dataset.py`,
  `evaluate_existing_testset.py`, etc.)
- **No persistent test suite** — testing was done by running scripts manually
- **No containerisation** — every developer needed Python, Node.js, and model weights installed
- **Flat JSON files** for knowledge graph storage — no deduplication, no incremental updates
- **No operational monitoring** — no way to detect when evaluation quality had degraded

The v1.1.0 development cycle transformed this into a production platform. The commit history
of 226 commits documents the journey.

---

## Milestone 1: Zero-Host-Dependency Initiative

**Goal:** Allow any engineer to run every development operation without installing anything
beyond Docker on their host machine.

**Phase 1 — Containerise the test runner:**  
Created `docker-compose.test.yml` and `Dockerfile.test` (two-stage: `deps` + `test`). The deps
stage installs all Python requirements; the test stage copies source and runs `pytest -n auto`.
Corporate proxy ARGs (`HTTP_PROXY`, `HTTPS_PROXY`) are accepted at build time via `--build-arg`
and passed through to `pip install` — critical for ASPEED's internal network environment.

**Phase 2 — Containerise ad-hoc tools:**  
Created `docker-compose.tools.yml` with 15 tool containers, all tagged with `profiles: ["tools"]`
so they never auto-start. Every tool reuses the existing `rag-eval:dev` image via bind-mount —
zero additional Dockerfiles.

**Phase 3 — Containerise initialisation:**  
Created `docker-compose.init.yml` for one-shot operations: DB schema migrations, model
preloading, and hash recomputation. These containers use the `init` profile and exit after
completion.

**Result:** A developer on a fresh machine can run the entire platform with three commands:
```bash
docker compose -f docker-compose.init.yml --profile init up
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml up -d
docker compose -f docker-compose.test.yml run --rm test
```

---

## Milestone 2: The Legacy Purge — Deleting 200+ Files

**Commit:** `35b101100`  
**Rationale:** A codebase with ~200 "dead" files is not maintainable. Session-progress Markdown
files, one-off fix scripts, and debug utilities that were never deleted erode confidence in
what the "real" codebase is.

### What was deleted and why

| Category | Count | Reasoning |
|----------|-------|-----------|
| Session-progress `.md` files (`NEXT_STEPS_PRIORITY_Vx`, `SESSION_COMPLETE.md`, …) | 99 | These are dev notes, not documentation. A `DEVELOPER_GUIDE.md` + `RELEASE_NOTES.md` are the correct artifacts. |
| `eval-pipeline` one-off scripts (`debug_*.py`, `fix_*.py`, `apply_*.py`) | 42 | These scripts were written to fix specific issues that no longer exist. The fixes were merged into the main codebase. |
| Root-level orphaned `test_*.py` files | 27 | Superseded by the organised `eval-pipeline/tests/` suite with proper fixtures and CI integration. |
| Runtime artefacts (`rag_evaluation_results_*.json`, `*.xlsx`, `*.log`) | 20 | These are pipeline outputs, not source code. They belong in `outputs/` which is `.gitignore`d. |
| Exploratory PNG charts | 15 | Committed by accident; not documentation or test fixtures. |
| Legacy `eval-pipeline` Dockerfiles | 5 | `Dockerfile.optimized`, `Dockerfile.proxy-test`, etc. — consolidated into one canonical `Dockerfile.test`. |
| Detached sub-projects (`rr-data/`, `rr-eval/`, `rr-rag/`, etc.) | 5 | Prototype sub-projects that were never integrated and are not referenced by anything. |
| BFG rewrite artefacts | 1 | `.git.bfg-report/` and `bfg.jar` — not source, not needed in the tree. |
| Stale portal `node_modules` snapshot | 1 | Should never have been committed; `.gitignore` updated. |

### What was deliberately kept

The surviving codebase is **100% intentional**:
- `eval-pipeline/src/` — evaluation logic
- `eval-pipeline/tests/` — the 369 pipeline tests
- `insights-portal/src/` — frontend application
- `services/` — 11 microservice implementations + 364 service tests
- `ragas/` — the modified RAGAS library submodule
- `config/` — pipeline configuration templates
- `scripts/` — operational scripts (e2e smoke, tag image, validate compose)
- `docs/` — architecture decision records and deployment guides
- `deploy/` + `helm/` — production deployment manifests

---

## Milestone 3: GCR Metric — The Core Innovation

**Files:** `eval-pipeline/src/evaluation/graph_context_relevance.py`,  
`eval-pipeline/src/utils/graph_store.py`,  
`eval-pipeline/tests/test_graph_context_relevance.py`

**The problem being solved:** Standard RAGAS metrics (Faithfulness, ContextPrecision,
ContextRecall) and cosine-similarity scores fail to distinguish between:
- A retrieval of 5 semantically relevant but **structurally disconnected** chunks
- A retrieval of 5 chunks that form a **coherent topological path** through the knowledge graph

Both score identically on cosine similarity, but the second retrieval produces drastically better
LLM answers because the generator receives a coherent story rather than disconnected fragments.

**Design decisions made:**

1. **Why SQLite, not Neo4j?** The `GraphStore` Protocol was designed so the backend is swappable.
   SQLite was chosen for the default because it has zero external dependencies (Python stdlib),
   runs fine for the batch-evaluation workload (single-threaded reads, cached NetworkX graph),
   and keeps the developer setup to a single `.db` file. The `Neo4jGraphStore` stub already exists
   in `graph_store.py` for future migration.

2. **Why mean Jaccard, not max Jaccard for $S_e$?** The generator sees all retrieved chunks, not
   just the best one. Max Jaccard would reward "at least one relevant node," which is a weaker
   contract than "all retrieved nodes should be relevant." Mean Jaccard is stricter and better
   models the actual downstream risk.

3. **Why $\mu + 2\sigma$ for hub detection, not a fixed percentile?** A percentile-based approach
   (e.g., 95th percentile) always marks the top 5% as hubs, even in a regular graph where no
   node is structurally dominant. The $\mu + 2\sigma$ rule returns `P_h = 0.0` for regular graphs
   (all nodes have equal degree) and only fires when there is genuine positively-skewed degree
   concentration. This is the correct behaviour for domain-specific corpora.

4. **Why cache the full graph in memory?** The GCR evaluator is invoked many times per pipeline
   run (once per QA pair). Loading the full graph from SQLite on every call would be
   $O((N + E) \times \text{num\_pairs})$. The instance-level cache reduces amortised cost to a
   single $O(N + E)$ load followed by $O(|R| \cdot |Q|)$ per evaluation. The `invalidate_cache()`
   method is provided for incremental-ingestion workflows.

**TDD process:** The test suite in `test_graph_context_relevance.py` was written against a
specification *before* the implementation was complete. Five graph topologies were designed as
fixtures: perfect chain, fragmented (zero edges), hub-contaminated, empty retrieval, and
single-node. Each fixture verifies a specific algorithm invariant.

---

## Milestone 4: Tiktoken Offline Cache Fix

**File:** `eval-pipeline/global_tiktoken_patch.py`,  
**Root cause:** RAGAS internally imports `tiktoken` for token counting. In an offline container
environment (no internet access), tiktoken's default behaviour is to attempt downloading BPE
encoding files on first use, which fails immediately with a network error and causes RAGAS to
crash before any evaluation occurs.

**The fix — a two-layer approach:**

**Layer 1: Pre-warm the cache at image build time.**  
The `Dockerfile.test` downloads the `cl100k_base` and `o200k_base` BPE files into a cache
directory during the image build (when network access is available). At test runtime, the
cache directory is bind-mounted so tiktoken finds its files without network access.

**Layer 2: Comprehensive fallback module.**  
`global_tiktoken_patch.py` creates a fully-compatible `tiktoken` mock module as a Python
`types.ModuleType` and injects it into `sys.modules` *before* RAGAS is imported. The mock
implements the complete `tiktoken` public API:
- `get_encoding(name)` → returns `ComprehensiveFallbackEncoding`
- `encoding_for_model(model_name)` → maps GPT model names to encoding names
- `encode(text)` → returns a word-count-proportional token list (not exact, but sufficient for RAGAS's use of token counts in prompt length calculations)
- `decode(tokens)` → returns a placeholder string
- `encode_batch(texts)`, `decode_batch(token_lists)` — batch variants

**How it's applied:** The `conftest.py` at the repo root imports and calls
`apply_global_tiktoken_patch()` as its first action, before any test or RAGAS import can
trigger the real tiktoken module.

**Environment variables that reinforce the fix:**

```
TIKTOKEN_CACHE_ONLY=1       # tiktoken reads only from pre-warmed cache
TIKTOKEN_DISABLE_DOWNLOAD=1 # prevents any network fetch attempt
```

---

## Milestone 5: Drift Detection System

**Files:** `services/eval/drift/detector.py`, `store.py`, `scheduler.py`, `notifier.py`,  
`eval-pipeline/webhook_daemon.py`, `insights-portal/src/components/DriftMonitorBanner.tsx`

**The problem:** After evaluations run in production, the system had no mechanism to detect
whether evaluation quality was trending downward over time. Engineers would only notice
degradation when users complained.

**Design decision — Welch Z-test over PSI (Population Stability Index):**
Population Stability Index is the industry standard for distribution drift monitoring in
tabular ML. It was evaluated and rejected for this use case because:
- PSI requires binning continuous distributions — the GCR sub-scores are bounded `[0, 1]` but
  the appropriate bin boundaries aren't obvious without empirical data
- PSI is not direction-aware — it detects *any* distribution shift including improvements
- The Welch Z-test is direction-aware: $S_e$ and $S_c$ are flagged only when they *drop*,
  $P_h$ only when it *rises*. This is the correct semantic: an $S_c$ *improvement* should not
  trigger an alert.

**Design decision — Two-stage severity (`WARNING` before `DRIFTING`):**
Paging an on-call engineer for a single-metric alert at 3AM is a high false-positive risk.
The two-stage design requires two *independent* metrics to flag simultaneously for a `DRIFTING`
alert. Under a null hypothesis, this reduces false positive rate from ~5% to ~0.25%.

**Integration pattern:** The drift system is deliberately embedded in the existing Webhook
Daemon process rather than as a new microservice. The daemon already had process lifetime
management (FastAPI lifespan) and outputs directory access. Adding APScheduler + a 50-line
detector required zero new service definitions, zero new Docker images, zero new ports.

---

## Milestone 6: Frontend — Insights Portal Overhaul

**Files:** `insights-portal/src/` (53 TypeScript files, ~8,000 lines)

The Insights Portal was rebuilt from a simple CSV-viewer prototype into a full evaluation
observability platform:

### Major components added/rebuilt

| Component | Description | Lines |
|-----------|-------------|-------|
| `QAFailureExplorer.tsx` | Master-detail QA debugger with per-row metric badges and entity highlighting | ~400 |
| `DriftMonitorBanner.tsx` | Drift status polling banner with per-metric Z-score display | ~300 |
| `CytoscapeGraph.tsx` | Interactive KG visualiser with Cytoscape.js force-directed layout | ~250 |
| `AiInsightCard.tsx` | LLM insight display with animated gradient border during generation | ~150 |
| `insights/engine.ts` | Deterministic rule-based insights engine (zero LLM dependency) | ~100 |
| `textHighlighter.ts` | Two-tier entity highlighter (structured → heuristic fallback) | ~80 |

### Why a deterministic insights engine over LLM-generated insights?

The `generateInsights()` function in `engine.ts` is pure TypeScript — no API calls, no model
calls. This was a deliberate architectural choice:

- **Hallucination-proof by construction.** The insight strings are template literals. The
  "intelligence" is encoded as version-controlled if/else rules.
- **Zero latency.** Insights appear instantly when the page loads.
- **Deterministic for audit trails.** The same KPI values always produce the same insights —
  reproducible for compliance review.

The `AiInsightCard` component (which *does* call an LLM) is a separate opt-in feature for
narrative summaries. The deterministic engine provides the baseline analysis layer that always
works regardless of LLM availability.

---

## Milestone 7: AI Auto-Insights Implementation

**File:** `services/reporting/` (AI Insights endpoint at `POST /api/v1/insights/generate`)

**What it does:** Accepts evaluation KPIs, the pass/fail verdict, and metric deltas against
thresholds, constructs a structured prompt that teaches the LLM the semantics of $S_e$, $S_c$,
$P_h$, Faithfulness, and AnswerRelevancy, and returns a narrative explanation.

**LLM compatibility:** The OpenAI-compatible async client pattern supports:
- GPT-4o-mini, GPT-4o (OpenAI direct)
- Azure OpenAI (set `INSIGHTS_API_BASE_URL` to the Azure endpoint)
- Ollama (local LLM server, `http://localhost:11434/v1`)
- vLLM inference server (any model)

**Error handling decisions:**

| Condition | Response | Rationale |
|-----------|----------|-----------|
| `INSIGHTS_API_KEY` not set | `503 Service Unavailable` + actionable message | Silent failure would produce empty insight cards with no explanation. An actionable error is better UX. |
| Upstream LLM returns 4xx/5xx | `502 Bad Gateway` | The reporting service is a proxy; the correct HTTP semantic is 502. |
| `AuthenticationError` | `503` with specific message | Distinguish "you need to configure a key" from "the service is down." |

---

## Key Architectural Decisions (ADRs)

### ADR-001: SQLite over Neo4j for GraphStore

**Status:** Accepted  
**Context:** The GCR metric requires a persistent graph store for Knowledge Graph data.  
**Decision:** Default to `SQLiteGraphStore` with a `GraphStore` Protocol for future backend swapping.  
**Consequences:** Zero runtime dependencies; no graph-native query language (Cypher); suitable for corpora up to ~1M nodes on commodity hardware before the O(N+E) graph load becomes a bottleneck.

### ADR-002: Content-hash addressing for node deduplication

**Status:** Accepted  
**Context:** Document ingestion pipelines frequently re-process overlapping document chunks.  
**Decision:** Use SHA-256[:32] of trimmed content as the primary key.  
**Consequences:** Identical chunks are never stored twice; whitespace-only differences are normalised; 128-bit address space makes collisions negligible for any realistic corpus size.

### ADR-003: per-test `tmp_path` isolation over shared test fixtures

**Status:** Accepted  
**Context:** 733 tests run in parallel via pytest-xdist.  
**Decision:** Every test that touches the filesystem uses pytest's function-scoped `tmp_path`; no session-scoped mutable fixtures.  
**Consequences:** Zero inter-test state contamination; slightly higher fixture setup overhead (acceptable given the test suite runtime of ~4s for collection and fast per-test I/O on SQLite `:memory:`-equivalent paths).

### ADR-004: Deterministic insights engine over LLM-only insights

**Status:** Accepted  
**Context:** Dashboard insights could be generated by an LLM or by deterministic rules.  
**Decision:** Implement a deterministic rule engine as the primary insights layer; LLM narrative as optional `AiInsightCard` layer.  
**Consequences:** Zero hallucination risk in the primary insights layer; instant rendering; LLM layer requires API key configuration and is explicitly opt-in.

### ADR-005: Welch Z-test for drift, not PSI

**Status:** Accepted  
**Context:** Need to detect statistical degradation in GCR metric time series.  
**Decision:** Apply direction-aware Welch one-sample Z-test to each sub-score independently.  
**Consequences:** Direction-aware (improvements don't trigger alerts); interpretable (Z-score has a natural "standard errors" interpretation); requires only mean/stdev of baseline, not binned histograms.

---

## Known Technical Debt & Planned Improvements

| Item | Priority | Notes |
|------|----------|-------|
| GCR weight calibration study | High | Default $\alpha=\beta=0.4, \gamma=0.2$ are principled priors; domain-specific calibration via Kendall-$\tau$ against human-labeled retrievals would improve accuracy |
| Lazy neighbourhood graph loading | Medium | Current `_build_graph()` loads the full SQLite graph into memory; for >500K node corpora, replace with scoped SQL neighborhood queries |
| Log-transform hub detection | Medium | $\mu + 2\sigma$ on raw degrees is appropriate for approximately normal distributions; power-law degree distributions (common in large KGs) would benefit from log-transformed degree statistics |
| `DriftDetector` unit test for edge cases | Low | `min_baseline` boundary conditions and `INSUFFICIENT_DATA` path coverage could be expanded |
| Reviewer service Postgres migration | Low | Reviewer state uses SQLite by default; `db-migrate` init container supports Postgres but it is not the production default |
