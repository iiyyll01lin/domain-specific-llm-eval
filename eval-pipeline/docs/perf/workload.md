# Baseline Workload Profile / 基線工作負載概述

Version: 0.1  
Status: Draft  
Date: 2025-09-10  
Owner: Platform Engineering

---
## 1. Purpose / 目的
Define a reproducible baseline workload used for:  
- Processing latency SLO validation (TASK-015)  
- Evaluation pipeline performance baseline (TASK-102)  
- Load & scalability tests design (TASK-103)  

定義可重現之基線工作負載，用於：  
- 處理延遲 SLO 驗證（TASK-015）  
- 評估管線效能基線（TASK-102）  
- 負載與擴展性測試設計（TASK-103）  

## 2. Document Scope / 範圍
Covers documents corpus composition, token size distribution, concurrency levels, and RAG evaluation sampling parameters used to produce baseline metrics prior to optimization.
涵蓋：文件語料組成、token 大小分佈、併發層級、RAG 評估抽樣參數；用於優化前取得基線指標。

## 3. Corpus Definition / 語料定義
| Corpus Segment | Count | Avg Tokens | P95 Tokens | Notes                    |
|----------------|-------|-----------:|-----------:|--------------------------|
| short_facts    | 120   |        800 |      1,050 | FAQ / glossary           |
| medium_guides  | 60    |      4,500 |      5,800 | How-to procedural        |
| long_specs     | 20    |     22,000 |     28,000 | Technical specs          |
| giant_refs     | 5     |     48,000 |     50,000 | Reference manuals (edge) |

Token counts measured using tokenizer version: baseline-tokenizer v0 (same as TASK-015a output).  

## 4. Processing Workload Parameters / 處理工作負載參數
| Parameter            | Value                                    | Rationale                                        |
|----------------------|------------------------------------------|--------------------------------------------------|
| max_chunk_tokens     | 512                                      | Align with embedding context economy             |
| chunk_overlap_tokens | 50                                       | Preserve semantic continuity                     |
| embedding_batch_size | 16                                       | Prevent API saturation (mid QPS)                 |
| parallel_docs        | 4                                        | Conservative concurrency to avoid I/O bottleneck |
| retry_backoff        | exp( base=0.5, factor=2, max=4 retries ) | Align with TASK-015c circuit breaker design      |

## 5. Evaluation Sampling Baseline / 評估抽樣基線
| Parameter               | Value | Notes                             |
|-------------------------|-------|-----------------------------------|
| questions_per_doc (avg) | 3     | Derived from initial RAGAS config |
| personas                | 4     | Distinct perspective coverage     |
| scenarios               | 3     | Context variation                 |
| random_seed             | 42    | Deterministic reproducibility     |

## 6. Concurrency & Traffic Model / 併發與流量模型
| Phase                | Concurrent Ops        | Description                         |
|----------------------|-----------------------|-------------------------------------|
| ingestion_processing | 4 documents in-flight | Matches parallel_docs parameter     |
| testset_generation   | 1 job                 | Avoid noise for baseline            |
| evaluation_run       | 2 threads             | Balanced vs metric computation load |
| websocket_clients    | 3                     | Simulated UI observers              |

## 7. Target Metrics / 目標指標
| Domain     | Metric                                | Target                              | Notes                          |
|------------|---------------------------------------|-------------------------------------|--------------------------------|
| Processing | Single 50k token doc p95 latency      | < 30s                               | TASK-015 SLO                   |
| Processing | All docs batch p95 latency            | < 25s                               | Excludes giant_refs edge cases |
| Evaluation | End-to-end run (100 Q/A) p95 duration | < 420s                              | Pre-optimization               |
| WebSocket  | Avg reconnect time                    | < 2s                                | TASK-070                       |
| WebSocket  | Heartbeat miss tolerance              | 2 consecutive misses (15s interval) | Trigger downgrade after 3rd    |

## 8. Data & Measurement Method / 測量方法
- Time origin: ingestion job accepted timestamp.  
- Processing latency: (document.processed event time - ingestion accepted).  
- p95 computed via HDR histogram library; warm-up (first 3 docs) excluded.  
- Reconnect time: disconnect to onOpen (client timestamp).  
- Heartbeat interval validation: server emits every 15s ±1s; client gap detector counts misses.

## 9. Reproducibility Checklist / 重現檢查表
- Fixed random_seed across all generation phases.  
- Identical tokenizer & embedding model versions pinned.  
- No background load except defined concurrency levels.  
- Clear logs: include doc_id, size_tokens, stage timings.  

## 10. Change Control / 變更控制
Any modification to workload parameters must:  
1. Update this document with version bump.  
2. Regenerate baseline metrics (TASK-102).  
3. Re-run load tests (TASK-103) if material impact expected.  

## 11. Open Items / 未決事項
| Item                      | Description               | Owner | Status     |
|---------------------------|---------------------------|-------|------------|
| tokenizer version hashing | Formal hash publication   | TBD   | Open       |
| dynamic scaling probe     | Auto adjust parallel_docs | TBD   | Evaluating |

