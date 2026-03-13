# Limitations Progress 2026-03-13

This note revisits the limitations listed in [README.md](/data/yy/domain-specific-llm-eval/README.md) and records what has now been improved in code, what remains limited, and why.

## Original Limitation 1

`Binary Keyword Presence Check`

Previous state:
- Keyword evaluation degraded to substring presence in fallback mode.
- Synonyms like `caliper` versus `thickness gauge` were not credited unless upstream extraction normalized them exactly.
- Partial terminology overlap such as `surface inspect` versus `surface inspection` was undervalued.

Implemented improvement:
- [contextual_keyword_evaluator.py](/data/yy/domain-specific-llm-eval/eval-pipeline/src/evaluation/contextual_keyword_evaluator.py) now supports:
  - exact matches
  - synonym-aware matches via configurable `keyword_synonyms`
  - partial lexical similarity scoring
  - optional semantic segment scoring when sentence-transformer models are available
- The evaluator now emits structured `mandatory_match_details`, `optional_match_details`, and `keyword_relevance_score` instead of only binary pass/fail output.

Current remaining limit:
- Synonym coverage is still config-driven rather than learned from a domain ontology.
- Semantic keyword scoring still depends on local embedding model availability and quality.

## Original Limitation 2

`RAGAS Dependency on References`

Previous state:
- Reference-based metrics could pass even when domain keywords were only weakly aligned with the answer wording.
- The no-reference side of evaluation was not consistently aggregated back into normalized evaluator output.

Implemented improvement:
- The contextual evaluator now exposes `evaluate_responses()` for normalized RAG payloads used by the main evaluation path.
- This provides a no-reference `keyword_relevance_score` aggregate alongside contextual keyword score aggregation.
- `evaluate_testset()` was fixed so expected and mandatory keywords are merged correctly instead of calling `evaluate_response()` with an invalid argument shape.

Current remaining limit:
- The final orchestration layer still reports RAGAS metrics and contextual metrics separately in several legacy flows rather than enforcing a single universal ranking formula.
- Some older diagnostic scripts outside maintained test paths still use legacy result schemas.

## Practical Impact

What improved immediately:
- Better resilience when exact keyword phrasing does not match the answer.
- More faithful domain scoring for bilingual or terminology-shifted responses.
- Cleaner integration path from contextual keyword evaluation back into normalized RAG response evaluation.

What still needs engineering:
- Repo-wide migration of legacy scripts to the maintained pytest suite.
- Broader typing rollout beyond the focused core modules already covered by `mypy.ini`.
- More explicit aggregation rules between contextual relevance, RAGAS, multimodal metrics, and human review outcomes.