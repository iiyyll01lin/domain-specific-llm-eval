# Priority Next Steps for Domain-Specific LLM Evaluation - Phase 2

We have successfully executed the first set of enhancements (CI/CD integration, Streamlit telemetry dashboard, skeleton custom metrics, and multi-hop structure preparation). Here are the next most important steps to tackle in order to mature the pipeline out of beta:

## 1. Mypy Strict Type Fixing (`High Priority`)
- **Action**: Currently, Mypy was rolled out, but the system generates missing type hint errors for `PipelineTelemetry` and dynamic parameter mapping in JSON dumps. We need to systematically go through `eval-pipeline/src/utils/pipeline_telemetry.py` and enforce exact dict structures with `TypedDict` and proper `Optional` types to ensure runtime stability.

## 2. Implement the Custom Metric Heuristics (`High Priority`)
- **Action**: In `ragas_evaluator.py`, we stubbed `DomainRegexHeuristic`. We need to wire this metric directly into the RAGAS execution lifecycle so that domain-specific regex constraints (e.g. checking for mandated phrasing) alter the RAGAS `score` penalty dynamically at runtime.

## 3. Local Model Memory Management / Caching (`Medium Priority`)
- **Action**: Enhance `HuggingFaceEmbeddings` caching logic. Right now, it works locally but doesn't persist efficiently across container resets. Implementing `langchain` local storage caching or leveraging a local Redis index for embedding lookups will lower the execution time radically for repeated CSV structures.

## 4. Multi-hop Context-Aware LLM Synthesizing (`Medium Priority`)
- **Action**: We added the `[ENHANCED SYNTHESIS RULE V2]` hook into `run_pure_ragas_pipeline.py`. The actual logic needs to be constructed that makes asynchronous API calls to the local LLM, asking it to verify overlap between nodes in the Knowledge Graph, mapping semantic correlation explicitly.

## 5. End-to-End Orchestration UI (`Low Priority`)
- **Action**: The Streamlit dashboard currently just views JSON telemetry. We should expand it to be an orchestrator UI that allows users to trigger `testset_generation` from the browser, effectively enabling zero-code RAG evaluation for standard end-users.
