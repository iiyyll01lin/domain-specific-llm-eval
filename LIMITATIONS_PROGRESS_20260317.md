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

### Temporal Causality Metrics Now Participate In Main RAGAS Evaluation

`TemporalCausalityEvaluator` is now merged into `RagasEvaluator.evaluate(...)` so temporal reasoning signals can contribute to the formatted metrics payload when timeline-style inputs are present.

Impact:

- this closes one of the gaps where a later-roadmap evaluator existed but was not actually in the main execution path
- regression coverage exists in `eval-pipeline/tests/test_ragas_evaluator_domain_metrics.py`

### Topology And Federated Outputs Are Richer And More Auditable

The repo now emits more useful artifact content for two previously-thin partial areas:

- force-graph topology payloads now include isolated nodes, high-centrality nodes, weak cluster groupings, node degrees, and graph density
- federated aggregation now supports tenant trust policy filtering plus audit-log events for submit/spool/replay actions

## Remaining Practical Limitations

These are still real limitations in the current repository and should not be overstated.

1. External-backend roadmap items are still only partially closed.
   Examples: full distributed execution, full marketplace trust service, real post-quantum cryptography, real cloud orchestration runtimes, and design-only V14 concepts still require infrastructure or product decisions outside this repo.

2. Human feedback still does not ingest actual reviewer labels by default.
   The queueing and policy logic are improved, but the repo still mostly recommends and persists review work rather than closing the loop with a maintained reviewer UI/backend.

3. Several advanced evaluators remain heuristic-heavy.
   Symbolic, spatial, intent, swarm, and temporal metrics are wired into the main path, but they still rely on lightweight local logic rather than robust backend-capable engines.

4. Legacy root-level test scripts still exist.
   More of their intent is now preserved in maintained pytest coverage, but the long tail of old `eval-pipeline/test_*.py` scripts has not been fully retired yet.

5. Repo-wide “all roadmap items complete” is still not an honest claim.
   The audit should continue to distinguish between implemented, partial, stub, and design-only items.

## Recommended Next Engineering Steps

1. Continue migrating remaining `eval-pipeline/test_*.py` scripts into `eval-pipeline/tests/` and then retire the old scripts in batches.
2. Add a persisted reviewer-results ingestion path so human feedback recommendations become a full learning loop.
3. Replace heuristic advanced evaluators with backend adapters only where deterministic fixtures and regression coverage can be maintained.
4. Keep V14 items design-gated until a reproducible simulator or scoring contract exists.