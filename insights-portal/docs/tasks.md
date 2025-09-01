# Insights Portal — Implementation Plan (Tasks)

This document tracks the implementation plan for Option A (Local-first SPA, React + Vite + TypeScript). Default locale is zh-TW with en-US toggle. Sample data location: `eval-pipeline/outputs/run_20250709_160725_85a5ba54/evaluations-pre`.

---

## 0) Overview & Principles
- References: `requirements.md` (EARS) and `design.md` (architecture/sequence/models/strategies).
- Goal: MVP for local/offline interaction, KPIs and gates, QA failure explorer, distribution analytics, export (CSV/XLSX), i18n toggle. PDF/PNG snapshot and advanced analytics come later.
- Cadence: Each task includes description, outcomes, deps/resources, EARS mapping, status (Planned/In-Progress/Done), and DoD.

---

## 1) Milestones
- M1 Scaffold & Env running
- M2 Loader & Parser & Normalization
- M3 Gates & Verdict
- M4 Dashboards v1 (Executive/QA/Analytics)
- M5 Export (CSV/XLSX) & i18n
- M6 Multi-run compare & Insights v1
- M7 Tests coverage & perf baseline

---

## 2) Task List (trackable)

### T-001 Scaffold & Environment
- Description: Install Node.js LTS, dependencies, run dev server; verify skeleton launches.
- Outcomes: `npm install` succeeds; `npm run dev` starts; landing shows nav and language switch.
- Deps/Res: Windows/PowerShell, Node.js LTS; `insights-portal/`.
- EARS: NFR (modern browsers/local-first).
- Status: In-Progress (scaffold, routes, i18n, profile load in place; manual dev run verification pending)
- DoD: Loads in browser without critical console errors.

### T-002 Tooling & Checks
- Description: Wire ESLint/Prettier/Vitest/Playwright; TS strict on; verify path aliases.
- Outcomes: `npm run lint` and `npm run test` pass initially; add minimal tests.
- Deps/Res: `package.json`, `tsconfig.json`.
- EARS: Reliability (clear errors), NFR.
- Status: In-Progress (ESLint/Prettier/Vitest included; strict TS enabled; minimal tests pending)
- DoD: CI/local one-shot lint/test green (minimal set).

### T-010 File Access (v1 single JSON)
- Description: File picker + permissions; show chosen base path.
- Outcomes: Interactive file selection; friendly errors on denied permission.
- Deps/Res: Browser FS Access API.
- EARS: Story 1 (Run discovery).
- Status: Done (v1 loads a single JSON with progress/errors; extended directory scan covered by T-011)
- DoD: After selecting file, parsing runs with progress and KPIs/row counts appear.

### T-011 Run discovery & artifact detection
- Description: Iterate runs under selected directory; detect supported files:
  - `ragas_enhanced_evaluation_results_*.json`
  - `config.yaml` (optional)
- Outcomes: List artifacts and counts; non-blocking warnings when missing.
- Deps/Res: T-010.
- EARS: Story 1 (artifact listing & counts).
- Status: Done
  - Implemented: pick directory and scan runs; detect result JSON and `config.yaml`; one-click load.
  - Added: artifact type counts (results/csv/other json); fast-scan returns total items and metrics coverage (tooltip shows full list).
  - Added: polished empty-state when no compatible files found.
  - TODO: Inline full metrics list beyond tooltip (optional polish).
- DoD: Each run displays artifact types and counts; shows “No compatible files” when none.

### T-012 JSON/CSV parsing & normalization (with latency stats)
- Description: In Worker, parse JSON/CSV via PapaParse chunked; map latency fields; compute avg/p50/p90/p99; return to UI.
- Outcomes:
  - Worker supports `parse-summary-json` and `parse-csv` and returns `latencies` (avg/p50/p90/p99).
  - `normalizeItem/aggregateKpis/computeLatencyStats` implemented; Executive Overview shows p50/p90.
- Status: In-Progress (CSV pipeline parses and computes; UI entry is still JSON-first and CSV flow not fully wired into UI)
- DoD: Can load `testset_with_rag_responses_*.csv`, show KPIs and latency stats; error messages include filename and row offsets (follow-up).

### T-012 Parser Worker & Zod schema validation
- Description: Web Worker performs parsing (JSON/CSV via PapaParse chunked), validates with Zod, outputs normalized structures.
- Outcomes: Emits `RunMeta` and `EvaluationItem[]`; incremental progress; clearer errors with filename/offset.
- Deps/Res: T-011; `design.md` models; `requirements.md` file patterns.
- EARS: Story 1, 13 (error details).
- Status: In-Progress (JSON worker, normalization, KPI aggregation done; supports `{items:[]}` or array input)
- DoD: ~5k rows within acceptable time on sample; broken file yields clear error messages.

### T-013 Metric normalization & missing-value policy
- Description: Unify metric keys and decimals; use null/N.A for missing; keep extended fields in `extra`.
- Outcomes: UI agnostic to raw differences; unknown metrics handled by registry.
- Deps/Res: T-012.
- EARS: Story 9 (metric extensibility).
- Status: Done (latency alias candidates implemented; unknown metrics pass-through)
- DoD: Unknown metric key appears as default KPI/distribution without custom code.

### T-020 Metrics Registry (extensible)
- Description: Schema-driven registry with label keys, i18n, default visual types, formatters; fallback for unknown keys.
- Outcomes: Minimal registry in place (labelKey, formatter); overview renders via registry and localized number formatting.
- Deps/Res: T-013; i18n.
- EARS: Story 9.
- Status: In-Progress (default visuals/help texts pending; generic card for unknown keys WIP)
- DoD: Unseen metric keys auto-render as KPI card and histogram (generic style).

### T-030 Threshold profile load & edit
- Description: Load `profiles/thresholds.standard.json`; allow editing `warning/critical` in UI.
- Outcomes: Threshold panel view/edit; reset to defaults or config.yaml overrides.
- Deps/Res: T-020.
- EARS: Story 3 (configurable release gates).
- Status: Done (App loads profile on start; RunDirectoryPicker merges `config.yaml` overrides; ThresholdEditor supports reset and live editing)
- DoD: Changing Faithfulness 0.30→0.40 immediately affects verdict.

### T-031 Verdict engine (rules evaluation)
- Description: Evaluate rules like `any(metric < critical)`, `any(metric < warning)`, `all(metric >= warning)`; output verdict plus triggered rules and failing metrics.
- Outcomes: Ready/At Risk/Blocked; rules configurable.
- Deps/Res: T-030.
- EARS: Story 2, 3.
- Status: Done (`evaluateVerdict` wired into Executive Overview; failing metrics displayed)
- DoD: Same data with different thresholds yields consistent verdict and highlights.

### T-040 Filters/Cohorts & aggregation engine
- Description: Filters by language/success flags/metric ranges/latency ranges; aggregate KPIs/distributions in Worker; 300ms (≤5k) redraw.
- Outcomes: Consistent KPIs/charts/table updates; active chips and one-click clear.
- Deps/Res: T-012, T-020.
- EARS: Story 5, 11 (perf).
- Status: In-Progress → Updated → UI Complete (MVP) → Enhanced (Dev timings + tunables + benchmarks + visualization)
  - Implemented: Global filters (language, latency range, metric ranges) in store; worker-side aggregate with filters; Executive Overview/QA/Analytics respect filters.
  - Added: FiltersBar with metric range sliders (0–1) and active chips with per-chip clear and Clear All; chips appear in QA and Overview; slider input debounced.
  - Added: Debounced aggregation in Overview (~200ms) and worker-side coalescing of aggregate requests (~100ms, adjustable via Dev panel) to reduce churn.
  - Added: Aggregation timings (filter/sample/aggregate) emitted by worker; Overview shows a Dev timings panel and sends sampling hint (>20k rows) to balance responsiveness and accuracy.
  - Added: DevTelemetryPanel with adjustable coalescing window and one-click benchmarks for 5k/20k/100k datasets; summaries by size × sampling% × coalesceMs; worker supports runtime `config` message to set coalesceMs. Added inline trend sparkline and grouped box summaries per dataset size; CSV export of benchmark results.
  - Added: For 100k+ datasets (no sampling), worker switches to chunked aggregation (10k slices) to keep UI responsive; timing stats still reported.
  - Tests: Added unit tests for metric range filter and chips clear behavior.
  - TODO: Performance tune for 20k+ (additional worker batching), polish labels and help texts.
  - TODO (next): Batch-run multiple coalesce×sample×size combinations and render comparison matrix (box/median lines), then export a consolidated report.
- DoD: ≤5k rows update within 300ms; 20k within 1s.

### T-050 Executive Overview
- Description: KPI cards, Verdict banner, warning/critical gaps; collapsible panels with persisted state.
- Outcomes: One screen to judge release readiness and gaps.
- Deps/Res: T-031, T-040.
- EARS: Story 2.
- Status: In-Progress (wired to RunLoader/DirectoryPicker; KPI cards + Verdict banner + counts + latency p50/p90; “sort by threshold gap” now sorts KPI cards by severity)
- DoD: With default thresholds, verdict renders; sub-threshold metrics show colors and gaps.

### T-051 QA Failure Explorer
- Description: Virtualized table with search/sort/pagination; row click opens details (≤200ms).
- Outcomes: Quickly locate low-scoring samples; support bookmarking and export.
- Deps/Res: T-040.
- EARS: Story 8.
- Status: In-Progress → Updated (virtualized + bookmarks + columns/persist) → Enhanced (prefs module + SLA test + details drawer)
  - Implemented: Sort by low scores, keyword search (question), row details panel; Export CSV/XLSX of current table with metadata.
  - Added: Lightweight virtualized table (windowed rendering) and bookmarking (star toggle) with XLSX export of bookmarks; persistent bookmarks (localStorage); selectable base columns and togglable metric columns.
  - Added: Persist visible column preferences (localStorage) for base and metric columns via prefs helpers.
  - Added: Row Details drawer panel with context expansion and lightweight highlighting; for long contexts, use chunked slicing (500 characters per slice) with a "Show more" progressive loading strategy to avoid excessive DOM; the SLA utility validates first open ≤200ms (typical contexts).
  - Tests: Added unit test for bookmark flag in exported rows; unit test for bookmarks persistence structure; added SLA utility/test for row details (≤200ms for immediate loader).
- DoD: Details show user_input/reference/rag_answer/contexts/metrics; bookmarks exported to CSV.

### T-052 Analytics Distribution (hist/box/scatter)
- Description: Histograms, box plots, scatter (e.g., Faithfulness vs AnswerRelevancy); multi-run legends.
- Outcomes: Visualize distributions and relations; interactive filters stay in sync.
- Deps/Res: T-040.
- EARS: Story 4 (DA view).
- Status: In-Progress → Updated (hist + box + scatter + CSV/XLSX/PNG) → Improved (two-way sync + outliers + grouped boxes)
  - Implemented: Histogram, Box plot, and Scatter with brush selection that updates global metric ranges; respects global filters; exports CSV/XLSX of source values and PNG snapshot.
  - Added: Scatter brush supports merging multiple areas (union of min/max) before updating global filters; axes reflect current metric range filters (two-way sync). Box plot displays outliers (1.5×IQR) and grouped box plots by cohort (language, success/failure, or failing metric bucket), including outliers.
  - TODO: Grouped comparison and multi-run legends in a later milestone.
- DoD: Range sliders reflect immediately; scatter enables brush to filter.

### T-060 Multi-run compare
- Description: Select 2–5 runs; show metric deltas (abs/%), highlight regressions over thresholds; missing metrics show N/A.
- Outcomes: Quickly identify improvements/regressions.
- Deps/Res: T-011, T-040.
- EARS: Story 6.
- Status: Planned
- DoD: Two-run comparison correctly shows differences and warnings.

### T-070 Export (CSV/XLSX)
- Description: Export current view table (visible columns, filters, row count); overview exports KPIs/thresholds/verdict.
- Outcomes: CSV/XLSX with metadata footer (run IDs, filters, timestamp, thresholds).
- Deps/Res: T-050, T-051, T-052.
- EARS: Story 10.
- Status: In-Progress → Updated → PDF plan scaffolded (Option B API draft)
  - Implemented: Exporter utility (CSV/XLSX via SheetJS); Executive Overview exports KPIs (CSV/XLSX, with metadata); QA Failure Explorer exports visible table (CSV/XLSX) with metadata; Analytics view exports CSV/XLSX of source values and front-end PNG snapshot.
  - Added: Branding/meta support in exports (CSV commented header/footer, XLSX 'branding' sheet) for title/brand/footer text; added PDF manifest builder (Option B pipeline plan) to define sections/meta; created `server/pdf-service.js` scaffold exposing `/render/pdf`, returning a tiny dummy PDF (with %PDF header). Added a lightweight golden test to check the header (`server/__tests__/pdf_service.test.ts`). Added Dockerfile and simple deploy script for the PDF service scaffold.
  - TODO: Styling templates and header/footer layout polish; wire Option B renderer（Puppeteer/Chromium）with Docker deployment; add golden fixtures/content assertions.
- DoD: CSV/XLSX are consumable with complete columns and metadata.

### T-080 i18n & a11y
- Description: zh-TW/en-US text and number/date localization; basic a11y (keyboard nav/contrast/alt text).
- Outcomes: Live language switch; charts and key components provide alt text or summary.
- Deps/Res: Global UI; LangSwitcher.
- EARS: Story 12.
- Status: In-Progress (i18n wired with LangSwitcher and persisted locale; added aria/alt on charts and controls; more a11y pending)
- DoD: No reload on language switch; RTL out of scope.

### T-081 Dark mode switch (default dark)
- Description: Provide a UI toggle to switch between Dark/Light themes; persist preference in localStorage; default to Dark on first load.
- Outcomes: Global theme applies to body, text, buttons, and form controls via CSS variables; toggle available in header next to language switcher.
- Deps/Res: T-080 (global UI), store.
- EARS: Story 12 (a11y — contrast and user preference).
- Status: Done (theme store, global stylesheet, and ThemeSwitcher wired; default dark, persists across reloads)
- DoD: Page loads in Dark by default; switching to Light updates immediately and persists after refresh.

### T-090 Error handling & resilience
- Description: Errors include filename/offset; missing metrics show N/A; memory pressure hints; no network calls by default offline.
- Outcomes: Self-healing UX without blocking main flows.
- Deps/Res: T-012; UI notifications.
- EARS: Story 1, 13, 14.
- Status: In-Progress (RunLoader/Worker/DirectoryPicker show errors; richer offsets/recovery and memory detection to add)
- DoD: Broken files reproducibly yield clear messages; N/A doesn’t break layout.

### T-100 Session save/load
- Description: Save/restore selected runs, filters, thresholds, persona, locale (small JSON).
- Outcomes: Reproduce same KPIs, filters, and verdict.
- Deps/Res: Global state.
- EARS: Story 15.
- Status: Planned
- DoD: After load, UI and numbers match.

### T-110 Insights (rules-based Top 3)
- Description: Heuristic/rule-based insights (with evidence: metrics, distributions, counts, sample links) with confidence scores.
- Outcomes: Top 3 recommendations and actions on overview or dedicated panel.
- Deps/Res: T-040, T-052.
- EARS: Story 7; PRD Addendum (Secs 8/9/10/11) start with rules.
- Status: Planned
- DoD: Contexts like low Faithfulness yield relevant suggestions and actionable steps.

### T-120 Tests coverage
- Description:
  - Unit (Vitest): Parser, Gates, Aggregation, Registry
  - Integration (RTL): Run loading, threshold changes, compare view
  - E2E (Playwright): QA table interactions, ≤200ms details, export with metadata
  - Perf: 5k/20k/100k dataset timings and Worker timings
- Outcomes: Stable green on baseline; perf targets met.
- Deps/Res: Feature modules.
- EARS: Multiple (11 perf, 8 failure explorer, 10 export).
- Status: Planned → Partially addressed (unit + integration smoke)
  - Added: Unit tests for QA preferences, row details SLA utility, and PDF manifest builder; worker parse/aggregate integration smoke test; PDF service golden header test.
  - Pending: Playwright E2E (QA interactions, row details ≤200ms, exports with branding/threshold metadata) and automated perf baseline scripts (integrated with the Dev panel). Playwright config and a disabled-by-default SLA E2E spec scaffolded.
  - DoD: CI shows main paths green; perf gates met or degradations explained; worker parse/aggregate integration and Playwright E2E pending.

### T-130 Docs & User Guide
- Description: Update `README.md`, add guide (select directory, set thresholds, export); troubleshooting (permissions/bad files).
- Outcomes: Non-technical users can self-serve.
- Deps/Res: Major features ready.
- EARS: Democratize results.
- Status: In-Progress (README zh/EN created; user guide and troubleshooting pending)
- DoD: New users analyze and export within 5 minutes.

### T-140 Option B backlog
- Description: FastAPI service skeleton and endpoints (PDF/PNG export, >100k pre-aggregation).
- Outcomes: Seamless expansion if needed; share JSON schema with frontend.
- Deps/Res: After v1.
- EARS: NFR scalability.
- Status: Planned
- DoD: Minimal health and export draft API docs.

---

## 3) Requirements Mapping
- Story 1: T-010, T-011, T-012
- Story 2: T-050, T-031
- Story 3: T-030, T-031
- Story 4: T-052
- Story 5: T-040
- Story 6: T-060
- Story 7: T-110
- Story 8: T-051
- Story 9: T-020
- Story 10: T-070
- Story 11: T-040, T-120
- Story 12: T-080
- Story 13: T-012, T-090
- Story 14: T-090
- Story 15: T-100

---

## 4) Risks & Mitigations
- Large datasets (>100k): incremental loading and sampling; disable heavy charts (T-040).
- Heterogeneous files: Zod tolerant parsing + adapters; preserve unknown fields (T-012/013).
- Misconfigured thresholds: provide reset and delta vs defaults (T-030).
- Export fidelity: start with CSV/XLSX; PDF/PNG via snapshots or Option B (T-070/T-140).

---

## 5) Recommendations
- Dashboard “condensed mode” and KPI pinning to reduce visual noise (T-050 scope).
- Insights panel supports bilingual copy for cross-team communication (T-110 + T-080).
- Consider PWA or Electron for better file-picking UX and offline packaging (Backlog).

---

## 6) Open Questions
- Baseline run for regression thresholds? (ties to T-060/T-031)
- Persona overrides import/export needed? (T-020 extensions)
- Export style requirements (branding header/footer templates)? (T-070)

---

## Progress Update — Current (2025-08-31)
- Done:
  - Verdict engine v1 (T-031); Executive Overview MVP shows KPIs, verdict, counts and latency p50/p90 (partial T-050).
  - Directory scan detects runs, artifact type counts, and items/metrics fast-scan (partial T-011).
  - Threshold profile load and `config.yaml` override merge (T-030).
  - i18n and LangSwitcher with persisted locale (partial T-080).
  - Single-file JSON load with Worker parsing, KPI aggregation, and latency stats (partial T-010/T-012).
  - Dark mode switch with default dark and persistence (T-081).
  - Filters engine with UI (language/latency/metric sliders + chips) wired in Overview/QA/Analytics (T-040 UI complete, engine ongoing perf work).
  - QA Failure Explorer: virtualized list, bookmarks with export, sort/search/details, export CSV/XLSX (T-051/T-070 expanded).
  - Analytics: histogram/box/scatter with brush that updates global filters; honors filters; CSV/XLSX export and PNG snapshot (T-052/T-070).
  - Run discovery coverage tooltip and polished empty-state (T-011 update).
  - Debounced sliders in FiltersBar, debounced Overview aggregation, and worker-side coalescing of aggregate requests (T-040 update).
  - Tooling solidified: ESLint/Prettier/Vitest configured; minimal unit tests added (T-002/T-120 partial).
- Added:
  - Overview “sort by threshold gap” toggle with active sorting (T-050).
  - Dev autoload sample JSON via /@fs for quicker verification.
- Next:
  - Optional polish for T-011: inline full metrics list beyond tooltip.
  - Filters perf: tune debounce values and batching for 20k+; measure worker timings (T-040).
  - QA Table: persist visible column preferences; row details perf SLA test (T-051/T-070).
  - Analytics: scatter multi-area brush merge, box outliers/groups, slider sync polish (T-052).
  - Export: styling templates and PDF option (Option B) (T-070/T-140).
  - Expand unit/integration/E2E tests (RTL/Playwright) (T-120).

Note: All code comments must be in English.

---

## Addendum — Data ingestion compatibility (2025-08-27)

Context: Files in `evaluations-pre/` are not a stable portal-ready shape; many are reports without per-item metrics. Portal now has best-effort extraction, but we should formalize the contract.

Action items (adopted):
- Pipeline: emit a portal-ready summary JSON per run under `outputs/<run-id>/portal/` with `{ items: [...] }`.
- Converter: added `eval-pipeline/convert_to_portal_summary.py` to transform `evaluations-pre` artifacts into the summary (averages metric scores grouped by question).
- Portal worker: keep current fallback for `items/results/evaluations/records/rows/data` (one-level nesting) — no further heuristics unless a concrete dataset requires it.
- Samples: maintain `public/samples/` for quick manual verification; include at least one array-shaped and one object-with-items file.
- Validation: optional CI step to assert that emitted summaries are parseable and produce non-zero items for the supported metrics.
