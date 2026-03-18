# Limitations Progress 2026-03-17

This note records the limitations that still remain after the latest code pass and the concrete improvements now implemented in the repository.

## Improvements Implemented In This Pass

### Human Feedback Selection Is No Longer Fixed-Threshold Only

The maintained `HumanFeedbackManager` now supports:

- dynamic uncertainty-range selection using score IQR with configurable bounds
- diverse confident-answer sampling for QA coverage
- smoothed feedback-variance tracking and adaptive review window sizing
- persisted feedback policy state in `feedback_policy_state.json`
- threshold recommendation output derived from recent score distributions

Impact:

- the queueing path now does more than simple low-confidence filtering
- the active-learning loop is closer to the README design intent
- regression coverage exists in `eval-pipeline/tests/test_human_feedback_manager.py`

### Reviewer Workflow Now Has Maintained UI/API And State Store

The maintained reviewer flow now supports:

- persisted reviewer-label ingestion through normalized records
- maintained FastAPI endpoints for listing and submitting review decisions
- Streamlit reviewer queue interaction in the telemetry dashboard
- SQLite-backed review state storage behind an explicit repository layer
- compatibility JSONL snapshots for existing artifact consumers

Impact:

- reviewer-approved or rejected items can now be reloaded in later runs
- resolved feedback items can suppress duplicate review requests
- reviewer scores and notes can flow back into maintained evaluation results
- reviewer workflow is no longer only a file-drop bridge
- regression coverage exists in `eval-pipeline/tests/test_human_feedback_manager.py`, `eval-pipeline/tests/test_webhook_daemon.py`, and `eval-pipeline/tests/test_reviewer_state_repository.py`

### Reviewer Workflow Now Has A Service Boundary Contract

The maintained reviewer workflow now also exposes a service boundary abstraction with:

- reviewer token authentication
- explicit reviewer identity handling
- tenant-scope enforcement
- moderation policy checks for review submission notes
- an optional HTTP client path for remote service deployment later

Impact:

- the repo no longer couples reviewer workflow only to in-process state management
- auth, identity, tenancy, and moderation rules can be exercised before choosing a remote DB or deployment target
- regression coverage exists in `eval-pipeline/tests/test_reviewer_service.py` and `eval-pipeline/tests/test_webhook_daemon.py`

### Reviewer Service Now Has A Standalone Deployable API Surface

The repo now also provides a standalone reviewer service entrypoint and deployment contract with:

- a separate FastAPI reviewer-service app
- health endpoint support for process-level deployment smoke tests
- an explicit deployment contract for auth headers, reviewer identity, tenancy, moderation decisions, and persistence schema
- a dedicated reviewer-service Dockerfile for isolated service rollout

Impact:

- reviewer workflow no longer depends on the webhook daemon as its only HTTP surface
- a minimal remote deployment contract now exists before introducing a production database or auth provider
- regression coverage exists in `eval-pipeline/tests/test_reviewer_service_api.py`

### Temporal Causality Metrics Now Participate In Main RAGAS Evaluation

`TemporalCausalityEvaluator` is now merged into `RagasEvaluator.evaluate(...)` so temporal reasoning signals can contribute to the formatted metrics payload when timeline-style inputs are present.

Impact:

- this closes one of the gaps where a later-roadmap evaluator existed but was not actually in the main execution path
- regression coverage exists in `eval-pipeline/tests/test_ragas_evaluator_domain_metrics.py`

### Topology And Federated Outputs Are Richer And More Auditable

The repo now emits more useful artifact content for two previously-thin partial areas:

- force-graph topology payloads now include isolated nodes, high-centrality nodes, weak cluster groupings, node degrees, and graph density
- federated aggregation now supports tenant trust policy filtering plus audit-log events for submit/spool/replay actions

### Hardware-Acceleration Observability Is Now Persisted And Displayed

The maintained `vLLMInferenceClient` now emits capability, benchmark, and runtime observability telemetry, and the orchestrator persists that telemetry into evaluation metadata plus a dedicated observability artifact.

Impact:

- accelerated inference configuration now has measurable latency / throughput artifacts
- capability snapshots can be stored alongside evaluation metadata
- latency percentile, error-mode, fallback-path, and GPU saturation views can now be rendered in the maintained dashboard
- regression coverage exists in `eval-pipeline/tests/test_v9_components.py`, `eval-pipeline/tests/test_pipeline_integration_regression.py`, and `eval-pipeline/tests/test_dashboard_data.py`

### Deterministic Backend Contracts Now Have Fixture Coverage

The maintained symbolic, spatial, intent, temporal, and swarm backend adapters now have explicit contract fixtures for accepted inputs and edge cases.

Impact:

- scoring drift is less likely to slip in unnoticed
- backend behavior is documented by executable fixture tables instead of only ad hoc assertions
- regression coverage exists in `eval-pipeline/tests/test_deterministic_backends.py` and `eval-pipeline/tests/test_swarm_contracts.py`

### Observability Now Has Cross-Run Retention Summaries

The maintained dashboard data layer now persists a cross-run observability retention index with:

- retention-window summaries
- recent-run comparison windows
- cross-run error drill-down aggregation
- artifact linking for telemetry and observability outputs

Impact:

- observability is no longer limited to per-run snapshots only
- the dashboard can compare recent windows and drill into retained artifacts
- regression coverage exists in `eval-pipeline/tests/test_dashboard_data.py`

### Failure Forensics Now Extend The Retention Layer

The retained observability layer now also includes:

- per-run diffs against the previous retained run
- error-mode to artifact cross-links
- retained latency and fallback anomaly flags

Impact:

- retained telemetry is more actionable for debugging regressions across runs
- error drill-down can now point to relevant retained artifacts directly
- regression coverage exists in `eval-pipeline/tests/test_dashboard_data.py`

### Additional Legacy Root Scripts Have Maintained Coverage Or Deprecation Markers

Maintained pytest coverage now also preserves core intent from additional root scripts, while stale root scripts now carry deprecation markers pointing at maintained tests.

Impact:

- `test_pipeline_fixes.py` core CLI intent is preserved in maintained coverage
- `test_custom_llm_integration.py` core config/document-processing intent is preserved in maintained coverage
- the repo is gradually moving away from print-driven ad hoc scripts toward maintained regression coverage

### Additional Legacy Runtime Smoke Tests Have Maintained Coverage

Maintained pytest coverage now preserves the core intent of additional legacy scripts including:

- `test_report_fixes.py`
- `test_tiktoken_patch.py`
- `test_full_ragas_implementation.py`

Impact:

- these checks no longer depend only on ad hoc root-level scripts
- maintained coverage exists in `eval-pipeline/tests/test_legacy_runtime_smoke_regressions.py`

### Fourth Legacy Migration Batch Has Started

Maintained pytest coverage now also preserves core intent from additional ad hoc scripts focused on document chunking, custom document loading, and comprehensive pipeline fixes.

Impact:

- more print-driven or subprocess-oriented root scripts now have maintained coverage anchors
- the long tail of root scripts is shrinking in practical risk even before full retirement
- maintained coverage exists in `eval-pipeline/tests/test_legacy_runtime_smoke_regressions.py`

### Mock RAGAS Results Are Now Explicitly Marked As Mock

The RAGAS bypass path now marks generated fallback results as mock outputs instead of letting them blend silently into normal result flows.

Impact:

- downstream consumers can distinguish mocked fallback metrics from real RAGAS outputs
- missing mock metrics now raise explicit key errors instead of silently returning empty arrays
- regression coverage exists in `eval-pipeline/tests/test_ragas_bypass_regression.py`

## Remaining Practical Limitations

These are still real limitations in the current repository and should not be overstated.

1. External-backend roadmap items are still only partially closed.
   Examples: full distributed execution, full marketplace trust service, real post-quantum cryptography, real cloud orchestration runtimes, and design-only V14 concepts still require infrastructure or product decisions outside this repo.

2. Human feedback now has a maintained workflow, service boundary, and standalone API surface, but it is not yet a production-grade deployed service.
   The repo now provides reviewer UI/API, SQLite-backed state storage, auth/identity/tenant/moderation service contracts, an optional remote client path, and a standalone reviewer-service app, but it still lacks persistent production storage, managed auth integration, service orchestration, and operational hardening.

3. Several advanced evaluators are now contract-backed, but they are still deterministic local engines.
   Symbolic, spatial, intent, temporal, and swarm metrics now use backend adapters with fixture coverage, but they still do not represent fully externalized reasoning services or model-backed arbitration engines.

4. Legacy root-level test scripts still exist.
   More batches of their intent are now preserved in maintained pytest coverage, but the long tail of old `eval-pipeline/test_*.py` scripts has not been fully retired yet.

5. Repo-wide “all roadmap items complete” is still not an honest claim.
   The audit should continue to distinguish between implemented, partial, stub, and design-only items.

## Recommended Next Engineering Steps

1. Continue migrating remaining `eval-pipeline/test_*.py` scripts into `eval-pipeline/tests/` and then retire the old scripts in batches.
2. Replace the current SQLite-backed reviewer persistence adapter behind the standalone reviewer-service API with a production persistence backend only after tenancy, auth source, and audit requirements are finalized.
3. Continue replacing heuristic evaluators with backend adapters only where deterministic fixtures and regression coverage can be maintained, prioritizing remaining mock or inconsistent fallback paths.
4. Expand hardware observability from current retained summaries and failure forensics into longer-horizon retention, richer anomaly classification, and artifact searchability.
5. Keep V14 items design-gated until a reproducible simulator or scoring contract exists.