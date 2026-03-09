# Insights Portal (Local-first SPA)

## Quick Start

1. Install dependencies
2. Start the dev server
3. Open http://localhost:5173 in your browser

Key concepts:
- Executive Overview: KPIs + Verdict + Insights
- QA Explorer: Low-scoring items, details, bookmarks
- Analytics: Distributions (hist/box/scatter) + brush-to-filter
- Compare: Multi-run statistical deltas & cohort comparison

Primary actions:
1. Select Directory → scan runs → Load
2. Adjust Thresholds → verdict & KPI gaps update immediately
3. Apply Filters (language / latency / metric ranges) → all views recompute via worker
4. Bookmark failing items in QA → Export bookmarks
5. Use Analytics brush to constrain metric ranges interactively
6. Add additional runs (Directory Picker “Add to Compare”) → open Compare to analyze deltas
7. Generate exports (CSV/XLSX/PNG) with metadata (run IDs, filters, thresholds, timestamp)
8. Save Session → later Load Session to restore filters, thresholds, locale, persona, selected runs
9. Review Insights panel for rule-based actions (hallucination risk, relevancy vs faithfulness gap, keyword issues)

## Provide sample data

- Use the sample folder in this repo:
  - `eval-pipeline/outputs/run_20250709_160725_85a5ba54/evaluations-pre`
- Once on the page, click "Select Directory" and choose that folder to load the data.

> This scaffold includes: React + Vite + TypeScript, Zustand, i18n with language switcher, and three basic pages (Executive Overview / QA / Analytics).

# Portal-ready Summary Generation and UI Test

This guide shows how to generate a portal-ready summary JSON from evaluation artifacts and test it in the Insights Portal UI.

## Prerequisites
- Python 3.8+
- Node.js 18.18+ / 20.x LTS (for the portal UI)
- A run folder with `evaluations-pre/` (e.g., `eval-pipeline/outputs/<run-id>/evaluations-pre`)

> Note: Node.js 12 is too old for the current TypeScript/Vite toolchain and will fail during `npm run build`.

## Generate a portal-ready summary JSON
- The converter aggregates per-question metrics from `ragas_enhanced_detailed_calculations_*.json` and produces `{ items: [...] }`.

```bash
python3 /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/convert_to_portal_summary.py \
  /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/outputs/<run-id>/evaluations-pre
# Output: /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/outputs/<run-id>/portal/ragas_enhanced_evaluation_results_<timestamp>_portal.json
```

Notes
- Metrics supported: ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerSimilarity, ContextualKeywordMean
- Snake_case variants are auto-mapped. Missing `id` falls back to common fields; per-item latency is optional.

## Test in the Insights Portal
1) Start the portal
```bash
cd /mnt/d/workspace/domain-specific-llm-eval/insights-portal
npm install
npm run dev
# Open http://localhost:5173
```

2) Optional: thresholds profile
```bash
mkdir -p public/profiles
cp -r ../profiles/* public/profiles/
```

3) Load the generated summary
- Click "選擇 JSON 檔載入 run" and choose the JSON in `outputs/<run-id>/portal/`.
- Or click "選擇資料夾並掃描 runs" and select `outputs/<run-id>/`, then click 載入.

Expected
- Verdict banner shows Ready/At Risk/Blocked; Items > 0
- KPI cards render metric values; latency p50/p90 shows N/A if latency is not provided
- Sorting by threshold gap works; editing thresholds updates verdict/gaps instantly
- Insights panel lists top triggered insights with evidence and actions when rules match.
- Session save/load JSON reproduces thresholds, filters, persona, selected runs.

## Troubleshooting
- 0 items: The source file may be a report-type JSON without per-item metrics. Use the converter against `ragas_enhanced_detailed_calculations_*.json` or adjust the pipeline to emit the summary.
- Missing metrics: Only supported keys are aggregated; add mapping if your pipeline uses different names.
- Latency: Add a `latencyMs` (or alias) field per item if you want latency stats in the UI.
- i18n: If text appears as translation keys (e.g., `compare.title`) ensure `i18n/index.ts` initializes before rendering (check console for init warnings in custom integrations/tests).
- Compare empty: Need ≥2 runs added (Directory Picker “Add to Compare”).
- High memory warning: Consider filtering, sampling (auto for >20k), or limiting heavy charts.

## CI suggestion (optional)
- Add a validation step to assert the emitted summary JSON parses and yields non-zero items for supported metrics.

---

## How to Test (UI / E2E / Unit)

1) Manual UI smoke
- Start dev server, then open `http://localhost:5173/?sample=run_minimal` to auto-load a sample.
- Navigate to QA view; confirm the table renders and opening the first row Details is responsive (target ≤200ms on typical datasets).
- Try filters (language/metric ranges), threshold edits, and exports on each page.

2) Unit/Integration (Vitest)
```bash
npm test
```

3) End-to-End (Playwright, gated)
- Install browsers (first time):
```bash
npx playwright install --with-deps
```
- Run all E2E (gated by env):
```bash
export PW_E2E_ENABLED=1
npm run test:e2e
```
- Run SLA-only spec (≤200ms details open):
```bash
export PW_E2E_ENABLED=1
npm run test:e2e:sla
```

Tips
- If SLA is flaky, run headed/debug: `npx playwright test e2e/qa-sla.spec.ts --headed --debug`.
- For production-like perf, use preview: `npm run build && npm run preview`, then point Playwright baseURL to the preview port.
- Stable UI selectors are exposed via `data-testid` (e.g., `qa-table`, `qa-row-<idx>`, `qa-details`).

## Session Management

Buttons in Executive Overview:
- Save Session → downloads JSON: { schemaVersion, runId, thresholds, filters, locale, persona, selectedRuns }
- Load Session → restore identical KPI calculations & verdict (deterministic for given inputs)

## Insights Engine (Rules-Based)
Current heuristics derive plain-language suggestions (English) referencing metric patterns. Example triggers:
- High ContextPrecision/Recall with low Faithfulness → hallucination risk
- Low ContextualKeywordMean variance → keyword coverage issue
- Large AnswerRelevancy vs Faithfulness gap → grounding consistency actions

## Accessibility & i18n
- zh-TW & en-US; toggle persists to localStorage key `portal.lang`.
- Chart containers expose `role="img"` and localized aria-labels.

## Export Metadata
All exports embed (where format permits):
- Timestamp, filters, thresholds, branding block (brand/title/footer)
- Multi-run compare: per-metric deltaAbs/deltaPct + sample counts + NA %

## Performance Targets
- ≤5k rows: filter recompute ≤300ms
- ≤20k rows: ≤1s with sampling hints
- >20k: adaptive sampling (25%) unless disabled; memory pressure warning if usage elevated.

## Roadmap (Excerpt)
- PDF/PNG high-fidelity (Option B service) — partial stub in `server/`
- Advanced correlation/regression analytics
- PWA/Electron packaging
- Optional LLM rationale generation (off by default; privacy-first design)

---

## Lifecycle Module (TASK-067 / TASK-081)

The `src/app/lifecycle/` module provides live pipeline-status views for the backend microservices.

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `window.ENABLE_KG_PANEL` | `false` | Show the KG Jobs panel with entity focus form |
| `window.ENABLE_WS_PANEL` | `false` | Show WebSocket live-status panel |

Set these in the browser console or via `public/config.js`:
```js
window.ENABLE_KG_PANEL = true
```

### KG Panel — Entity Focus (Subgraph)

When `ENABLE_KG_PANEL` is set, completed KG jobs display an inline **Entity Focus** form:

```
Seed node: [___________]  Depth: [2]  [Fetch Subgraph]
```

Submitting calls `POST /kg-jobs/{id}/subgraph` and opens a modal overlay showing:
- **Nodes list** with optional `type` label and ✂ truncation indicator
- **Edges list** (`source → relation → target`)
- **SamplingPill**: orange `Sampled N/total` pill when the backend sampled the subgraph;
  green `Full N` pill when not sampled

### Telemetry (TASK-081)

All high-frequency UI events are batched and flushed via `src/telemetry/logEvent.ts`.

```ts
import { logEvent, configureTelemetry } from '@/telemetry/logEvent'

// Configure once at app bootstrap:
configureTelemetry({ endpoint: '/api/telemetry/events', batchSize: 20, flushIntervalMs: 5000 })

// Log events:
logEvent({ type: 'ui.kg.render', payload: { nodeCount: 42, durationMs: 120 } })
logEvent({ type: 'ui.ws.connect' })
```

**Supported event types** (ADR-005 taxonomy):

| Event type | Triggered when |
|-----------|----------------|
| `ui.page.load` | App first renders |
| `ui.kg.render` | KG panel mounts |
| `ui.kg.subgraph.fetch` | Subgraph fetch succeeds |
| `ui.kg.subgraph.error` | Subgraph fetch fails |
| `ui.ws.connect` | WebSocket opens |
| `ui.ws.disconnect` | WebSocket closes |
| `ui.eval.run` | Eval run triggered from UI |

Events are flushed on batch fill, timer, or tab close (`visibilitychange`). Failures are silently discarded — telemetry **never** blocks user interactions.

### API Configuration

Backend URLs are read from `getLifecycleConfig()`:

```ts
// src/app/lifecycle/config.ts
{
  processingBaseUrl: 'http://localhost:8002',
  testsetBaseUrl:    'http://localhost:8003',
  evalBaseUrl:       'http://localhost:8005',
  reportingBaseUrl:  'http://localhost:8006',
  kgBaseUrl:         'http://localhost:8008',
  wsBaseUrl:         'http://localhost:8009',
  requestTimeoutMs:  30000,
}
```

Override at runtime via `window.__LIFECYCLE_CONFIG__`.


