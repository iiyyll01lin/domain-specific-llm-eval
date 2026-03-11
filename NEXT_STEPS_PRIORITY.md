# Priority Next Steps for Domain-Specific LLM Evaluation

After conducting a comprehensive test pass and code review, the pipeline is now stable, and the bugs blocking `pytest` execution have been resolved. The custom telemetry and semantic chunking mechanisms are working robustly. Based on the current architecture, here is the roadmap for the most important enhancements:

## 1. Domain-Specific Telemetry Dashboard (`High Priority`)
Now that `runtime_telemetry.json` robustly captures extraction rates, processing times, and synthesis errors step-by-step, the next logical step is to visualize this.
- **Action**: Create a Streamlit or Gradio UI that parses the telemetry outputs. This will allow non-technical reviewers to instantly spot bottlenecks or low-quality generation sources without reading JSON logs.

## 2. CI/CD Integration for Evaluation Validation (`Medium Priority`)
The pytest suite proves execution reliability. We should ensure these checks always run.
- **Action**: Implement a GitHub Actions workflow (`.github/workflows/eval-ci.yml`) to automatically execute the `evaluate_existing_testset.py` suite against down-sampled CSVs on every PR. This guarantees that model configurations and prompts don't regress.

## 3. Custom Evaluation Metrics & Rubrics (`Medium Priority`)
Currently, you are leveraging default RAGAS metrics (Faithfulness, Context Precision). You have established `eval-pipeline/src/evaluation/ragas_evaluator.py`, which is perfectly positioned for extension.
- **Action**: Introduce domain-specific heuristics (e.g., custom regex checks, specialized hallucination rules geared toward your dataset's specific terminology).

## 4. Enhanced RAGAS Synthesis Rules (`Low-Medium Priority`)
The local `all-MiniLM-L6-v2` via HuggingFace works great for initial similarity embeddings, but generation quality could be boosted.
- **Action**: Investigate integrating LLM context-aware synthesizers during the `create_knowledge_graph_from_documents` step to generate deeper multi-hop relationships.

## 5. Type Hinting & Linting Rollout (`Low Priority`)
We've applied `black` and `isort` for structural formatting, but static type adherence could be tighter.
- **Action**: Roll out `mypy` natively across the pipeline to strictly type the data classes traversing from CSV extraction into RAGAS evaluation instances, further bulletproofing the codebase against edge cases.
