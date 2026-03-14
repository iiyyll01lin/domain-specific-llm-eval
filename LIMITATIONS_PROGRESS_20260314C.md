# Limitations Progress 2026-03-14C

This note extends [LIMITATIONS_PROGRESS_20260314B.md](/data/yy/domain-specific-llm-eval/LIMITATIONS_PROGRESS_20260314B.md) after another implementation pass.

## What Improved In This Pass

1. Several `Partial` roadmap modules are now backed by file or HTTP adapters instead of in-memory-only behavior.

Code paths improved:
- [taxonomy_discovery.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/loaders/taxonomy_discovery.py)
- [force_graph_viewer.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/ui/force_graph_viewer.py)
- [app_store_marketplace.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/ui/app_store_marketplace.py)
- [federated_learning.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/distributed/federated_learning.py)
- [orchestrator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/pipeline/orchestrator.py)

Practical effect:
- Taxonomy discovery can now read CSV files, persist proposals, approve outputs, and optionally merge backend-provided ontology hints.
- Graph topology can now export payload and HTML artifacts from persisted KG JSON.
- App-store manifests can now be loaded from files with install receipts written to disk.
- Federated aggregation can now attempt HTTP submission and spool failed submissions locally.

2. More legacy integration smoke has been absorbed into maintained pytest.

New maintained regression:
- [test_pipeline_integration_regression.py](/data/yy/domain-specific-llm-eval/eval-pipeline/tests/test_pipeline_integration_regression.py)

Practical effect:
- Stable pieces of the older pipeline integration smoke are now covered in normal pytest runs, especially metadata/output helper behavior around taxonomy, topology, app-store, and federated submission hooks.

## Remaining Limits

1. Backend-backed does not yet mean production-complete.
Current state:
- The new adapters persist and submit data, but they still rely on lightweight local protocols, simple JSON contracts, and minimal policy logic.

2. Legacy test migration is still incomplete.
Current state:
- More smoke scripts have been migrated, but many standalone `eval-pipeline/test_*.py` files remain outside maintained `eval-pipeline/tests/` coverage.

3. Threshold stabilization and universal scoring are still not globally unified.
Current state:
- The evaluator path is richer than before, but the repo still lacks a single universally enforced final aggregation contract across all legacy and modern paths.

4. Several later-roadmap items remain heuristic-first.
Current state:
- Federated, swarm, symbolic, spatial, intent, taxonomy, and topology paths are more executable now, but still do not represent production-grade external runtimes.

5. External dependency churn still causes some warning noise.
Current state:
- Some upstream LangChain / multipart warnings remain in broader test runs and should be cleaned up separately from functional correctness work.

## Practical Next Targets

1. Keep migrating remaining standalone `eval-pipeline/test_*.py` scripts into maintained pytest coverage.
2. Push taxonomy and topology from artifact export into richer UI or approval workflows.
3. Upgrade federated and app-store adapters toward stronger policy, identity, and remote service integration.
4. Continue reducing non-functional warning noise from upstream dependency churn and older pipeline paths.