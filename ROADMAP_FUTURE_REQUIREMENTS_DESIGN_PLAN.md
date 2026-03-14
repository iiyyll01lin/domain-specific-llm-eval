# Roadmap Future Requirements, Design, and Implementation Plan

This document captures the items from the later roadmap phases that are not yet honest-to-goodness production features in the current repository.

The goal is to move them out of vague aspiration and into three concrete buckets:
- requirements
- design
- implementation plan

This file does not claim these systems are already implemented.

## Scope Selection

These are the roadmap areas that still require substantial real engineering or external infrastructure before they can be marked `Implemented` in [ROADMAP_COMPLETION_AUDIT.md](/data/yy/domain-specific-llm-eval/ROADMAP_COMPLETION_AUDIT.md):

1. Federated learning edge tiers
2. Multi-agent swarm synthesis
3. Hyperparameter search orchestration
4. Hardware-accelerated inference integration
5. Knowledge-graph 3D topology visualization
6. Quantum-resistant PII tokenization
7. Unified LLM application store
8. Zero-shot taxonomy discovery
9. Neuro-symbolic RAG engine completion
10. WikiData / external KG synchronization hardening
11. Native alignment training backend execution
12. Mixed-reality / spatial evaluation deepening
13. Temporal causality multi-agent scoring
14. IoT / swarm telemetry live integration
15. Post-language BCI scoring hardening
16. V14 design-only items

## Requirements

### Federated Learning Edge Tiers

Functional requirements:
- Support local metric aggregation from multiple workers with signed result envelopes.
- Preserve tenant isolation and PII boundaries.
- Allow offline result upload and later central aggregation.

Non-functional requirements:
- Deterministic aggregation.
- Replayable audit logs.
- Configurable trust policy for accepted worker submissions.

### Multi-Agent Swarm Synthesis

Functional requirements:
- Multiple expert agents must evaluate or refine a candidate answer.
- Each agent must emit a structured verdict instead of appending plain text.
- Final synthesis must include agreement rate, dissent reasons, and confidence.

### Hyperparameter Search

Functional requirements:
- Search over chunk size, relationship thresholds, and metric weights.
- Persist trial inputs and outcomes.
- Support offline replay and best-run export.

### Hardware Acceleration

Functional requirements:
- Support real vLLM / TensorRT-LLM endpoint configuration.
- Detect capability mismatch and fall back safely.
- Emit benchmarkable throughput / latency telemetry.

### KG 3D Topology Visualization

Functional requirements:
- Render actual exported relationship graphs.
- Highlight isolated nodes, high-centrality nodes, and weakly connected clusters.
- Export a browser-consumable topology payload.

### Quantum-Resistant PII Tokenization

Functional requirements:
- Stable tokenization for repeated values.
- Reversible or escrow-backed detokenization policy when allowed.
- Explicit cryptographic metadata in audit logs.

### Unified App Store

Functional requirements:
- Versioned runbook manifests.
- Install validation and dependency checks.
- Source authenticity and trust metadata.

### Zero-Shot Taxonomy Discovery

Functional requirements:
- Discover candidate entities, relations, and hierarchy from unlabeled corpora.
- Emit confidence-scored ontology proposals.
- Allow human approval before promotion into KG generation.

### Neuro-Symbolic / WikiData / Alignment / Spatial / Temporal / IoT / BCI / V14

Shared requirements:
- Must be testable with deterministic fixtures.
- Must not rely on hand-wavy placeholder scoring.
- Must define explicit interfaces for external systems before code generation starts.

## Design

### Common Design Principle

Every currently-nonfinal roadmap feature should first land as:
- typed interface
- config schema
- deterministic local fixture implementation
- regression test
- optional external backend adapter

### Example Design Patterns

Federated learning:
- `EdgeResultEnvelope`
- `FederatedSubmissionValidator`
- `FederatedAggregationCoordinator`

Swarm synthesis:
- `AgentVerdict`
- `SwarmDebateRound`
- `ConsensusComputation`

Hyperparameter search:
- `SearchSpace`
- `TrialResult`
- `OptimizationBackend`

Hardware acceleration:
- `InferenceEndpointCapabilities`
- `AcceleratedClient`
- `LatencyBenchmarkReport`

V14 items:
- stay design-first until a concrete simulator, scoring contract, and reproducible fixture set exist

## Implementation Plan

### Phase A: Convert Stubs to Real Local Contracts

1. Replace placeholder return values with typed payloads and deterministic computation.
2. Add maintained pytest coverage under `eval-pipeline/tests/`.
3. Add telemetry and config wiring for each feature.

### Phase B: Add External Backend Adapters

1. Add HTTP or subprocess adapters for real services.
2. Keep adapters optional and disable by default.
3. Add fixture-backed tests that do not require live network access.

### Phase C: Promote Audit Status

Only mark a roadmap item `Implemented` once:
- a real execution path uses it
- config can enable it
- regression tests cover it
- it no longer returns placeholder values for the main path

## Current Practical Priority Order

Completed in recent passes:
- Threat intelligence real API adapter
- Agent-RAG structured tool trace metrics
- GitHub benchmark comment automation
- WikiData sync hardening
- Alignment backend command execution
- Hyperparameter search persistence and local trial history
- Swarm synthesis structured verdicts
- Main-path symbolic / spatial / intent metric wiring
- Maintained regression migration for legacy RAGAS and pure-RAGAS smoke checks

Next practical priority order:

1. Continue legacy `eval-pipeline/test_*.py` migration into maintained pytest
2. Deepen taxonomy discovery beyond header/value heuristics
3. Add richer hardware capability and benchmark telemetry for accelerated inference
4. Promote graph topology payloads into a real UI runtime
5. Replace simulated tokenization/app-store flows with backend-backed implementations where appropriate
6. Move federated, swarm, and symbolic paths from evaluator hooks to richer end-to-end runtimes
7. Keep remaining design-only items gated behind concrete backend or simulator decisions