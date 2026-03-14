# Limitations Progress 2026-03-14

This update extends [LIMITATIONS_PROGRESS_20260313.md](/data/yy/domain-specific-llm-eval/LIMITATIONS_PROGRESS_20260313.md) after another implementation pass.

## What Improved In This Pass

1. Several future-roadmap placeholders are no longer pure stubs.

Code paths improved:
- [hyperparam_search.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/optimization/hyperparam_search.py)
- [swarm_agent.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/swarm_agent.py)
- [taxonomy_discovery.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/loaders/taxonomy_discovery.py)
- [quantum_pii_tokenizer.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/security/quantum_pii_tokenizer.py)
- [app_store_marketplace.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/ui/app_store_marketplace.py)
- [vllm_client.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/inference/vllm_client.py)
- [force_graph_viewer.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/ui/force_graph_viewer.py)
- [dspy_autocorrect.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/dspy_autocorrect.py)

Practical effect:
- These modules now expose structured local contracts, persistence, escrow or manifest metadata, or richer payloads instead of returning fixed demo values only.

2. More legacy diagnostics were absorbed into maintained pytest coverage.

New maintained regression:
- [test_query_distribution_regression.py](/data/yy/domain-specific-llm-eval/eval-pipeline/tests/test_query_distribution_regression.py)
- [test_url_handling_regression.py](/data/yy/domain-specific-llm-eval/eval-pipeline/tests/test_url_handling_regression.py)
- [test_output_parser_regression.py](/data/yy/domain-specific-llm-eval/eval-pipeline/tests/test_output_parser_regression.py)

Practical effect:
- Query-distribution tracker/import smoke checks plus URL normalization and output-parser stability checks are now covered by normal regression execution rather than only standalone scripts.

3. Federated aggregation is no longer a pure placeholder.

Code path improved:
- [federated_learning.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/distributed/federated_learning.py)

Practical effect:
- Edge results are now wrapped in signed envelopes and aggregated deterministically with weighted scores and tenant summaries.
- This is still a local contract, not a real distributed parameter-server deployment.

## Remaining Limits

1. Many upgraded future-roadmap modules are still `Partial`, not `Implemented`.
Why:
- They now have better local contracts, but several are still not wired into the primary pipeline or production service execution paths.

2. Threshold stabilization and cross-metric aggregation are still incomplete.
Why:
- README-level ideas like adaptive variance smoothing, parameter self-tuning, and universal score aggregation are still only partially realized across the codebase.

3. Semantic quality still depends on optional models and external backends.
Why:
- Contextual relevance, richer taxonomy discovery, and accelerated inference improve when extra backends or models are available; fallback paths are stronger now, but still lower fidelity.

4. Legacy script migration is still in progress.
Why:
- Several `eval-pipeline/test_*.py` scripts remain outside maintained `eval-pipeline/tests/` coverage.

5. The most speculative V12-V14 items remain design-bound.
Why:
- They still require simulator choices, external infrastructure, or explicit interface decisions before implementation would be technically honest.

## Current Next Best Engineering Targets

1. Continue migrating remaining legacy `eval-pipeline/test_*.py` scripts into maintained pytest.
2. Wire partial V8-V10 modules into real execution paths instead of leaving them as isolated local contracts.
3. Push threshold smoothing and universal score aggregation deeper into the main evaluation/reporting flows.
4. Keep later-phase items in requirements/design mode until concrete backends are selected.