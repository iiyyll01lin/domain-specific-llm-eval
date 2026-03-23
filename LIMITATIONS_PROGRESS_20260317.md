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

### Reviewer Service Now Has Configurable Auth Sources And Production Backend Wiring

The maintained reviewer service now supports:

- configurable reviewer principal and tenant-membership sources
- static token auth for local/dev use
- file-backed reviewer principal directories
- internal signed-token auth for service-to-service or delegated issuer flows
- PostgreSQL-backed reviewer state storage alongside the earlier SQLite adapter

Impact:

- reviewer principal resolution is no longer hard-wired only to a single local token path
- tenant membership can now come from an explicit reviewer principal source instead of only runtime defaults
- production persistence wiring is now available before a full managed deployment is introduced
- regression coverage exists in `eval-pipeline/tests/test_reviewer_auth.py`, `eval-pipeline/tests/test_reviewer_state_repository_postgres.py`, `eval-pipeline/tests/test_reviewer_service.py`, and `eval-pipeline/tests/test_reviewer_service_api.py`

### Reviewer Deployment Manifests Now Exist

The repo now also includes reviewer-service deployment scaffolding with:

- a dedicated compose file for reviewer-service + PostgreSQL + reverse proxy
- an environment sample for deployment variables
- readiness and liveness probes via `/readyz` and `/healthz`
- Kubernetes deployment and ingress examples
- reverse proxy header forwarding for reviewer auth context

Impact:

- reviewer service deployment is no longer only implicit or ad hoc
- DB DSN wiring, health probes, and ingress/proxy structure now have concrete repo artifacts
- the remaining work is operational hardening rather than first-pass deployment definition

### Internal Reviewer Tokens Now Have Issuance, Rotation, And Revocation State

The maintained reviewer auth path now also supports a real internal issuer workflow with:

- a dedicated token issuance service with persisted keyring state
- signing-key rotation with grace windows via `kid` tracking
- token revocation state keyed by `jti`
- reviewer-service issuer endpoints for issuance, rotation, revocation, and issuer health
- auth-source validation against keyring and revocation files instead of only a single shared secret

Impact:

- internal-token auth is no longer only a static signature verifier
- service-to-service reviewer auth can now be exercised with a realistic issuance lifecycle before integrating an external IdP
- regression coverage exists in `eval-pipeline/tests/test_reviewer_auth.py` and `eval-pipeline/tests/test_reviewer_service_api.py`

### PostgreSQL Reviewer Backend Now Has Migration, Audit, Backup, And Pooling Contracts

The maintained PostgreSQL reviewer backend now also supports:

- explicit schema migration state tracking
- connection-pool configuration hooks with TLS/connect-timeout DSN hardening
- an audit-log schema for queue upserts, replacements, and reviewer-result ingestion
- JSON backup / restore contracts for operational export and recovery drills
- a dedicated migration CLI entrypoint for repository state initialization

Impact:

- reviewer persistence is no longer limited to best-effort schema creation in application startup
- operational readiness now includes auditable state transitions and a reproducible backup surface
- regression coverage exists in `eval-pipeline/tests/test_reviewer_state_repository_postgres.py`

### Reviewer State Recovery Now Includes Backup Restore And Audit Replay Coverage

The maintained reviewer state layer now also supports:

- exact JSON backup restore into fresh state stores
- audit-log restoration alongside queue and reviewer-result payloads
- deterministic audit replay into clean repositories for recovery drills
- reviewer-result recovery carrying `tenant_id` and `created_at` metadata through serialization paths

Impact:

- reviewer recovery is no longer limited to queue/result payload import without audit continuity
- backup verification can now be exercised as an end-to-end restore/replay path rather than a write-only export surface
- regression coverage exists in `eval-pipeline/tests/test_reviewer_state_repository.py`

### Reviewer Deployment Hardening Has Advanced Beyond First-Pass Manifests

The reviewer deployment scaffolding now also includes:

- a Kubernetes `ServiceAccount`
- a Kubernetes `Service`
- a `NetworkPolicy`
- a secret example manifest for DSN and issuer bootstrap material
- startup probe support, container resource requests/limits, and container/pod security context hardening
- structured reverse-proxy access logs with request IDs

Impact:

- deployment assets now cover the minimum production-hardening surface instead of only basic liveness/readiness wiring
- reviewer-service ingress and runtime behavior are easier to operate and audit in cluster environments

### Forensics Searchable Index Now Exposes An Operator Workflow Surface

The retained observability layer now also exposes maintained helper paths for:

- artifact search and filtering
- severity-ordered triage queue generation
- run-to-run artifact diff views
- issue-cluster drill-down payloads

Impact:

- searchable forensics is no longer only a retained JSON artifact
- operator-facing review flows can now be built on a maintained API/helper surface rather than manual artifact inspection
- regression coverage exists in `eval-pipeline/tests/test_dashboard_data.py`

### Sixth Legacy Migration Batch Now Has Maintained Coverage Anchors

Maintained pytest coverage now also preserves core intent from additional root scripts focused on:

- direct orchestrator smoke behavior
- document processing smoke behavior
- local generation smoke behavior
- comprehensive report generation smoke behavior

Impact:

- the sixth migration batch closes more of the print-driven root-script tail the repo still carried
- maintained coverage exists in `eval-pipeline/tests/test_legacy_root_script_regressions.py`

### Fallback-Heavy Evaluators Now Share A More Explicit Result Contract

The maintained evaluator layer now also has a shared result-contract helper with:

- normalized `result_source`
- normalized `error_stage`
- normalized `mock_data`
- explicit `contract_version`

Impact:

- fallback-heavy evaluators are less likely to drift into incompatible result shapes
- reviewer, reporting, and downstream debugging paths can rely on a more stable provenance contract
- regression coverage exists in `eval-pipeline/tests/test_evaluator_result_contracts.py`

### Core RAGAS Evaluator Paths Now Emit The Normalized Result Contract

The maintained `RagasEvaluator.evaluate(...)` path now also normalizes:

- unavailable-library responses
- invalid-input responses
- dataset-creation failures
- mock fallback responses
- successful evaluation responses
- terminal exception responses

Impact:

- one of the most central fallback-heavy evaluators is no longer mixing ad hoc return shapes across major branches
- downstream pipeline consumers can rely on `result_source`, `error_stage`, `mock_data`, and `contract_version` for the core RAGAS path as well
- regression coverage exists in `eval-pipeline/tests/test_ragas_evaluator_domain_metrics.py`

### Seventh Legacy Migration Batch Has Started

Maintained pytest coverage now also preserves core intent from additional print-driven legacy scripts focused on:

- direct reporting smoke behavior
- gates integration smoke behavior
- enhanced evaluator initialization behavior

Impact:

- another batch of root-level debug and verification scripts now has maintained regression anchors
- maintained coverage exists in `eval-pipeline/tests/test_legacy_root_script_regressions.py`

### More Evaluator And Pipeline Surfaces Now Share The Normalized Result Contract

The maintained evaluation and orchestration layer now also normalizes more high-traffic result paths, including:

- `RAGEvaluator.evaluate_single_testset(...)` and `RAGEvaluator.evaluate_testsets(...)`
- `ComprehensiveRAGEvaluatorFixed.evaluate_testset(...)`
- `ComprehensiveRAGEvaluatorFixed._generate_comprehensive_report(...)`
- `PipelineOrchestrator` stage results for testset generation, evaluation, and reporting
- the older `ComprehensiveRAGEvaluator` contextual, RAGAS, and testset-level outputs

Impact:

- downstream pipeline stages are less dependent on ad hoc `success` and `error` shapes
- batch and stage failures now carry clearer provenance through `result_source`, `error_stage`, and `contract_version`
- regression coverage exists in `eval-pipeline/tests/test_rag_evaluator_regression.py`, `eval-pipeline/tests/test_evaluator_result_contracts.py`, and `eval-pipeline/tests/test_pipeline_integration_regression.py`

### Eighth Legacy Migration Batch Has Started

Maintained pytest coverage now also preserves core intent from additional print-driven legacy scripts focused on:

- detailed CSV pipeline verification behavior
- pure-RAGAS conversion and generation smoke behavior
- orchestrator initialization smoke behavior

Impact:

- more root-level verification scripts now have deterministic maintained regression anchors
- the remaining legacy test-script tail is smaller and better documented by executable tests
- maintained coverage exists in `eval-pipeline/tests/test_legacy_root_script_regressions.py`

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

### Failure Forensics Now Have A Searchable Artifact Index

The retained observability layer now also includes:

- anomaly severity classification
- artifact search keys
- run-to-run regression labels
- retained issue clustering

Impact:

- retained telemetry can now be searched and grouped instead of only browsed manually
- regressions across runs are easier to triage into stable, warning, and critical cohorts
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

### Fifth Legacy Migration Batch Has Started

Maintained pytest coverage now also preserves core intent from additional ad hoc scripts focused on config verification, report generation smoke checks, and orchestrator update verification.

Impact:

- more debug-heavy and verification-heavy root scripts are now covered by maintained regressions
- the migration is now covering pipeline verification and direct reporting style scripts, not only narrower utilities
- maintained coverage exists in `eval-pipeline/tests/test_legacy_runtime_smoke_regressions.py`

### Mock RAGAS Results Are Now Explicitly Marked As Mock

The RAGAS bypass path now marks generated fallback results as mock outputs instead of letting them blend silently into normal result flows.

Impact:

- downstream consumers can distinguish mocked fallback metrics from real RAGAS outputs
- missing mock metrics now raise explicit key errors instead of silently returning empty arrays
- regression coverage exists in `eval-pipeline/tests/test_ragas_bypass_regression.py`

### Additional Evaluation Fallback Paths Now Expose Source And Stage Metadata

The maintained evaluation path now exposes clearer result provenance for more fallback branches, including RAGAS evaluator fallback and disabled/mock paths.

Impact:

- downstream consumers can distinguish between model-dump-fix mock results, generic fallback mock results, and explicit disabled paths
- failure diagnosis is less dependent on log-only inspection
- regression coverage exists in `eval-pipeline/tests/test_ragas_evaluator_domain_metrics.py`

### Hardware Acceleration Path Now Attempts Real Inference Before Simulated Fallback

The maintained `vLLMInferenceClient` now attempts a real completion request before falling back to simulated responses when the backend is unavailable or generation fails.

Impact:

- hardware observability is less likely to be based on fabricated generation paths when a real backend is reachable
- fallback telemetry now more accurately reflects direct inference versus simulated response paths
- regression coverage exists in `eval-pipeline/tests/test_v9_components.py` and `eval-pipeline/tests/test_pipeline_integration_regression.py`

## Remaining Practical Limitations

These are still real limitations in the current repository and should not be overstated.

1. External-backend roadmap items are still only partially closed.
   Examples: full distributed execution, full marketplace trust service, real post-quantum cryptography, real cloud orchestration runtimes, and design-only V14 concepts still require infrastructure or product decisions outside this repo.

2. Human feedback now has a maintained workflow, configurable auth source, internal issuer lifecycle, production-backend wiring, operator-forensics helper surface, and hardened deployment scaffolding, but it is not yet a production-grade deployed service.
   The repo now provides reviewer UI/API, SQLite and PostgreSQL-backed state paths, configurable principal sources, internal issuance / rotation / revocation state, audit-log and backup contracts, standalone reviewer-service deployment manifests with baseline hardening, and an operator-facing forensics helper surface, but it still lacks managed secrets, managed auth integration, service orchestration, audited production rollout, and real operations automation.

3. Several advanced evaluators are now contract-backed, but they are still deterministic local engines.
   Symbolic, spatial, intent, temporal, and swarm metrics now use backend adapters with fixture coverage, but they still do not represent fully externalized reasoning services or model-backed arbitration engines.

4. Legacy root-level test scripts still exist.
   More batches of their intent are now preserved in maintained pytest coverage, including the sixth and seventh migration batches, but the long tail of old `eval-pipeline/test_*.py` scripts has not been fully retired yet.

5. Repo-wide “all roadmap items complete” is still not an honest claim.
   The audit should continue to distinguish between implemented, partial, stub, and design-only items.

## Recommended Next Engineering Steps

1. Continue migrating remaining `eval-pipeline/test_*.py` scripts into `eval-pipeline/tests/` and then retire the old scripts in batches.
2. Connect the reviewer service to a managed principal source or OIDC provider only after claim mapping, tenancy semantics, and audit logging requirements are finalized.
3. Continue replacing heuristic evaluators with backend adapters only where deterministic fixtures and regression coverage can be maintained, prioritizing remaining mock or inconsistent fallback paths.
4. Expand hardware observability from current retained summaries, searchable forensics, and anomaly labels into longer-horizon retention, richer clustering, and operator-facing triage workflows.
5. Keep V14 items design-gated until a reproducible simulator or scoring contract exists.