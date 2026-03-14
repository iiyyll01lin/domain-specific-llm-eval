# Limitations Progress 2026-03-14B

This note extends [LIMITATIONS_PROGRESS_20260314.md](/data/yy/domain-specific-llm-eval/LIMITATIONS_PROGRESS_20260314.md) after another implementation pass.

## What Improved In This Pass

1. More `Partial` roadmap modules now participate in a real primary evaluation path.

Code paths improved:
- [ragas_evaluator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/ragas_evaluator.py)
- [swarm_agent.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/swarm_agent.py)
- [neuro_symbolic_rag.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/generation/neuro_symbolic_rag.py)
- [symbolic_evaluator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/symbolic_evaluator.py)
- [spatial_rag_evaluator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/spatial_rag_evaluator.py)
- [telepathic_intent_evaluator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/telepathic_intent_evaluator.py)
- [orchestrator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/pipeline/orchestrator.py)

Practical effect:
- Swarm, symbolic proof, spatial reasoning, and EEG-intent alignment metrics are no longer trapped in isolated demo modules only.
- Hyperparameter search can now be triggered from the evaluation stage metadata path when enabled.

2. More legacy smoke scripts were absorbed into maintained pytest coverage.

New maintained regression:
- [test_ragas_legacy_regression.py](/data/yy/domain-specific-llm-eval/eval-pipeline/tests/test_ragas_legacy_regression.py)

Practical effect:
- Core checks from older RAGAS field-mapping, pure-RAGAS import/KG smoke, config/secrets smoke, and hyperparameter helper smoke are now part of the normal maintained test suite.

## Remaining Limits

1. Several roadmap features are now wired, but still only as heuristic or local adapters.
Current state:
- Swarm, symbolic, spatial, and intent metrics now run in the main evaluation path, but the underlying engines are still lightweight heuristics rather than production-grade runtimes.

2. Legacy test migration is still incomplete.
Current state:
- Important older scripts have been migrated, but many `eval-pipeline/test_*.py` files still remain outside maintained `eval-pipeline/tests/` coverage.

3. Some roadmap items still lack end-to-end backend execution.
Current state:
- Federated learning, app-store publishing, tokenization crypto hardening, and hardware acceleration remain partial because they still need real backend/runtime integration.

4. Threshold stabilization and universal score aggregation are still not fully unified.
Current state:
- The repo has stronger component metrics than before, but there is still no single final universal ranking contract applied consistently across all legacy and modern flows.

5. External dependency churn still creates non-functional warning noise.
Current state:
- The current `HybridTestsetGenerator` path still emits an upstream LangChain deprecation warning around embedding class usage until the repo fully migrates to the newer package/runtime combination.

## Practical Next Targets

1. Keep migrating legacy `eval-pipeline/test_*.py` scripts into maintained pytest until the standalone smoke inventory is exhausted.
2. Push taxonomy discovery and graph topology from local payloads into richer production-style execution paths.
3. Replace heuristic swarm/symbolic/spatial/intent adapters with backend-capable or policy-aware runtimes where technically justified.
4. Continue reducing warning noise from upstream dependency deprecations and old pipeline paths.