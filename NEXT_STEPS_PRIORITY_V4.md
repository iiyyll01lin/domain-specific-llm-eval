# Priority Next Steps for Domain-Specific LLM Evaluation - Phase 4

We have completely matured the codebase under `V3` specifications: local LLM caching with `SQLiteCache`, integrated metric weighting (Faithfulness + Domain Regex rules), embedded manual human-in-the-loop detection for generated datasets, and properly attached the Docker Compose volumes. 

Since all testing is passing 100% and core stability is achieved, here are the most important remaining domains to explore:

## 1. Cloud-Native Object Storage Unification (`High Priority`)
- **Action**: Currently, knowledge graphs and testsets output locally into nested timestamp directories. With our MinIO instances defined in `docker-compose.services.yml`, we need to wire `PipelineTelemetry` and the `.json` / `.csv` dumps directly to an S3-compatible Boto3 interface (MinIO) so output is centralized across scale.

## 2. Advanced Multi-Agent LLM Evaluators (`Medium Priority`)
- **Action**: The single `custom_llm` model handles synthetic generation and evaluation. Split this architecture by routing synthetics logic to an "Actor" chain, and evaluation/metric heuristics to a separate "Critic" LLM in `RagasEvaluator`, to prevent single-model bias padding the scores.

## 3. Streaming UI Dashboard Refinements (`Low Priority`)
- **Action**: The Streamlit interface executes `subprocess` commands sequentially blocking the browser. Convert the UI's trigger process into an async `Celery` or `FastAPI Background Task` runner so users can queue multiple configurations and see a live progressing generation bar tracking documents processed. 
