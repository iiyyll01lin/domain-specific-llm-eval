# Priority Next Steps for Domain-Specific LLM Evaluation - Phase 3

Having fully implemented the `V2` roadmap (strict type patching, customized DomainRegex heuristics, dynamic embedding caching resilience, simulated multi-hop linkage hooks, and Orchestration UI triggering from Streamlit), we are officially hitting the bounds of pipeline maturity. 

## 1. Local Caching Deep Integration (`High Priority`)
- **Action**: We implemented conditional caching (`CacheBackedEmbeddings`), but the true Langchain `InMemoryCache` or local SQLite vector cache for the LLM itself would heavily reduce repeating identical evaluation requests across large batches. Implement `langchain.cache.SQLiteCache` for testset evaluation loops.

## 2. Dynamic Metric Weighting (`Medium Priority`)
- **Action**: Currently, the `DomainRegexHeuristic` metric and Ragas' defaults run parallel. Build a custom orchestrator mechanism where `Faithfulness` could be weighted `0.7` and `Regex` `0.3` to compute a unified `domain_score`.

## 3. Human-in-the-Loop Edge Case Fixing (`Medium Priority`)
- **Action**: We have deterministic processing, but some synthesizers fail quietly natively. Implement an Argilla or LangSmith trace hook directly into the synthesis phase that tags generated test queries that fall below a certain length threshold for manual human review.

## 4. Docker Compose Environment Parity (`Low Priority`)
- **Action**: Now that local components dynamically degrade or catch errors smoothly, the existing `docker-compose.yml` needs an overhaul to mount `~/.cache` dynamically so the Docker container benefits from locally cached embedding models without redownloading weights on container reboot.
