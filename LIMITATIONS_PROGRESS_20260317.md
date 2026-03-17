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

### Reviewer Result Ingestion Now Exists

The maintained `HumanFeedbackManager` now supports persisted reviewer-label ingestion through normalized JSON/JSONL records.

Impact:

- reviewer-approved or rejected items can now be reloaded in later runs
- resolved feedback items can suppress duplicate review requests
- reviewer scores and notes can flow back into maintained evaluation results
- regression coverage exists in `eval-pipeline/tests/test_human_feedback_manager.py`

### Temporal Causality Metrics Now Participate In Main RAGAS Evaluation

`TemporalCausalityEvaluator` is now merged into `RagasEvaluator.evaluate(...)` so temporal reasoning signals can contribute to the formatted metrics payload when timeline-style inputs are present.

Impact:

- this closes one of the gaps where a later-roadmap evaluator existed but was not actually in the main execution path
- regression coverage exists in `eval-pipeline/tests/test_ragas_evaluator_domain_metrics.py`

### Topology And Federated Outputs Are Richer And More Auditable

The repo now emits more useful artifact content for two previously-thin partial areas:

- force-graph topology payloads now include isolated nodes, high-centrality nodes, weak cluster groupings, node degrees, and graph density
- federated aggregation now supports tenant trust policy filtering plus audit-log events for submit/spool/replay actions

### Hardware-Acceleration Telemetry Is Now Emitted

The maintained `vLLMInferenceClient` now emits capability and benchmark telemetry, and the orchestrator can persist that telemetry into evaluation metadata when hardware acceleration is enabled.

Impact:

- accelerated inference configuration now has measurable latency / throughput artifacts
- capability snapshots can be stored alongside evaluation metadata
- regression coverage exists in `eval-pipeline/tests/test_v9_components.py` and `eval-pipeline/tests/test_pipeline_integration_regression.py`

### Additional Legacy Runtime Smoke Tests Have Maintained Coverage

Maintained pytest coverage now preserves the core intent of additional legacy scripts including:

- `test_report_fixes.py`
- `test_tiktoken_patch.py`
- `test_full_ragas_implementation.py`

Impact:

- these checks no longer depend only on ad hoc root-level scripts
- maintained coverage exists in `eval-pipeline/tests/test_legacy_runtime_smoke_regressions.py`

## Remaining Practical Limitations

These are still real limitations in the current repository and should not be overstated.

1. External-backend roadmap items are still only partially closed.
   Examples: full distributed execution, full marketplace trust service, real post-quantum cryptography, real cloud orchestration runtimes, and design-only V14 concepts still require infrastructure or product decisions outside this repo.

2. Human feedback still lacks a maintained reviewer UI/backend.
   Reviewer labels can now be ingested from persisted files, but the repo still does not provide a first-class reviewer workflow, moderation interface, or active feedback service.

3. Several advanced evaluators remain heuristic-heavy.
   Symbolic, spatial, intent, swarm, and temporal metrics are wired into the main path, but they still rely on lightweight local logic rather than robust backend-capable engines.

4. Legacy root-level test scripts still exist.
   More of their intent is now preserved in maintained pytest coverage, but the long tail of old `eval-pipeline/test_*.py` scripts has not been fully retired yet.

5. Repo-wide “all roadmap items complete” is still not an honest claim.
   The audit should continue to distinguish between implemented, partial, stub, and design-only items.

## Recommended Next Engineering Steps

1. Continue migrating remaining `eval-pipeline/test_*.py` scripts into `eval-pipeline/tests/` and then retire the old scripts in batches.
2. Add a maintained reviewer UI/backend so reviewer-result ingestion becomes a complete workflow instead of a file-based bridge.
3. Replace heuristic advanced evaluators with backend adapters only where deterministic fixtures and regression coverage can be maintained.
4. Expand hardware telemetry from local benchmark snapshots into richer runtime observability and failure analysis.
5. Keep V14 items design-gated until a reproducible simulator or scoring contract exists.