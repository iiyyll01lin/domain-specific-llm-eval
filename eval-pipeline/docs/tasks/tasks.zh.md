# 實作計畫 – RAG 評估平台與 UI 生命週期主控台

版本：0.1  
狀態：規劃草稿  
日期：2025-09-10  
負責：平台工程組  

---
## 1. 目的
將需求 (`requirements.zh.md`、`requirements.ui.zh.md`) 與設計文件 (`design.zh.md`、`design.ui.zh.md`) 轉化為可追蹤、可驗證之工程工作項目。涵蓋後端微服務轉型與嵌入式 UI Lifecycle Console（位於 `insights-portal/`）整合。

## 2. 指導原則
- 追溯優先：每個 FR / UI-FR 皆至少對應 1 個任務。
- 漸進強化：先交付端到端最小骨幹（documents→report），再優化。
- 決定性：初期即落實冪等/雜湊錨點，為快取與重現性鋪路。
- 內建可觀測：第一個可執行切片即具備結構化日誌與基本 metrics。
- 功能旗標隔離：KG 視覺化、多執行比較等為延遲載入與旗標控制。

## 3. 里程碑（指標性）
| 里程碑                             | Sprint | 交付項                                                             |
|------------------------------------|--------|--------------------------------------------------------------------|
| M1 基線 Ingestion→Processing       | 1      | Ingestion + Processing 服務、Artifacts、UI Documents/Processing 表格 |
| M2 Testset & Evaluation 骨幹       | 2      | Testset Gen、Evaluation Runner、評估 artifacts、UI Testsets / Runs    |
| M3 報告與 KM 摘要                  | 3      | Reporting、run_meta 連結、KM 匯出摘要與 UI                           |
| M4 知識圖譜（旗標）                  | 4      | KG builder、summary endpoint、UI KG 摘要 + 視覺化 lazy load          |
| M5 WebSocket 與效能強化            | 5      | WS multiplex、輪詢排程器、bundle 體積守門                            |
| M6 進階 Telemetry 與 Subgraph 草稿 | 6      | Telemetry taxonomy、Subgraph API draft、manifest 完整性原型          |

## 3.1 工程師分工與時間線
| 工程師                    | 角色焦點                                              | 彙總範圍                                                                |
|---------------------------|-------------------------------------------------------|-------------------------------------------------------------------------|
| E1 (後端骨幹)             | 服務骨架、Ingestion/Processing、儲存、效能/安全強化      | TASK-001~005, 010~016, 015a-d, 協助 033 I/O, 後續 090+                  |
| E2 (生成與評估)           | Testset 生成、Evaluation Runner、Metrics/聚合、Reporting | TASK-020a-d, 021a-d, 022~024, 030a-d, 031a-d, 032, 033a-d, 034, 040~043 |
| E3 (UI/即時/KG/Telemetry) | UI Panels、輪詢→WS、KG 旗標功能、Telemetry、Bundle 治理   | TASK-017, 025, 036, 044, 060+ (旗標), 070+, 081/082, 118/119            |

### 3.1.1 Sprint 級分配 (指標性)
| Sprint | 關鍵路徑 (主責)                                                 | 並行 (負責)                                                               | 備註                    |
|--------|-----------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------|
| 1      | Ingestion→Processing 主鏈 (E1: 001-005, 010-016, 015a-d)        | UI Documents/Processing (E3: 017); Testset 雜湊/驗證前置 (E2: 020a, 020c) | 建立決定性 & 可觀測基礎 |
| 2      | Testset 全流程 + Eval API (E2: 020b-d, 021a-d, 022-024, 030a-d) | Processing 強化 (E1: 015c/d); UI Testsets / Runs (E3: 025, 036)           | 問答→評估串接           |
| 3      | Metrics/聚合 + Reporting (E2: 031a-d, 032, 033a-d, 034, 035)    | KM 摘要支援 (E1); Reports & Summaries UI (E3)                             | 產出 kpis.json/報告     |
| 4      | KG 服務 (E3: 060-063, 062a-d；E1 協助抽取)                       | Summary 匯出整合 (E2: 045)                                                | 功能旗標避免阻塞        |
| 5      | WebSocket + 降級邏輯 (E3: 070-073)                              | Backpressure/Perf 調優 (E2: 033/034); 擴增 metrics (E1: 080)              | 從輪詢升級實時          |
| 6      | Telemetry taxonomy & Subgraph draft (E3: 066/067, 081, 118/119) | Manifest 完整性原型 (E2: 083)                                             | 治理與可觀測深化        |

> 備註：本檔案後續擴展任務請沿用 `engineer` 與 `target_sprint` 欄位標記模式。

## 4. 任務欄位定義
| 欄位                | 說明                    |
|---------------------|-------------------------|
| ID                  | TASK-### 唯一識別       |
| Title               | 精簡行動式標題          |
| Description         | 實作細節與理由          |
| Acceptance Criteria | 可驗證完成條件          |
| Dependencies        | 前置任務或資源          |
| Artifacts           | 產出或修改的檔案 / 端點 |
| Req Mapping         | 對應 FR / UI-FR / NFR   |

### 4.1 驗收條件範本（Acceptance Criteria Template）
以可測試、可量化為原則；每條至少能對應一個自動化檢查（單元 / 整合 / CI Gate）。

| 類型                 | 模板 Pattern                               | 範例                                                             | 量化要求                     |
|----------------------|--------------------------------------------|------------------------------------------------------------------|------------------------------|
| 功能 Functional      | Given <前置> When <操作> Then <可觀測結果> | Given 有效文件 When POST /documents Then 202 並回傳 job_id(UUID) | 指定 HTTP code + schema 欄位 |
| 冪等 Idempotency     | 重複 <操作> N 次 → 相同 <artifact/hash/id> | Hash 計算連續 3 次相同                                           | N≥2 且相等判斷               |
| 決定性 Determinism   | seed=<值> 前 N 產出穩定                    | seed=42 前 5 題一致                                              | 定義 N + seed                |
| 效能 Performance     | p95 < 閾值 (基線 workload)                 | 50k tokens 處理 p95 <30s                                         | 參照 workload.md 測量法      |
| 韌性 Resilience      | 注入 <故障> → 觸發 <重試/降級>             | 429 → 3 次指數退避後成功                                         | 次數 / backoff 上限          |
| 可觀測 Observability | 指標 <name> 發佈含 <labels>                | processing_embedding_batch_duration_seconds 存在                 | 指標名 + 至少 1 label        |
| 安全 / 供應鏈        | 工具 <scanner> 無 HIGH 退出碼 0            | Trivy 無 HIGH/CRITICAL                                           | 嚴重度閾值                   |
| 資料完整性           | <manifest/count/hash> 與產出一致           | chunk_count == embedding_count                                   | 精準欄位比對                 |
| 降級策略             | <N> 次失敗 → 降級於 T 秒內觸發             | 2 次心跳 miss → 5s 內降級                                        | N + 反應時間                 |

Checklist：
```
- [ ] 每條敘述符合模板
- [ ] 至少一條量化 (p95 / hash / count)
- [ ] 效能類連回 workload.md
- [ ] 有故障 / 負面路徑（若屬韌性任務）
	status: Completed
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	engineer: E2
	target_sprint: 3
	completed_on: 2025-10-02
	risk: "分位數計算錯誤"
	mitigation: "固定樣本測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_distribution.py -q
	deliverables:
		- services/eval/aggregation/distribution.py
		- services/tests/eval/test_distribution.py
	dod:
		- p50/p95 等分位數在固定樣本測試中符合期望
		- 支援負值與重複值運算且無精度漂移
		- 文件補充使用說明
# TASK-001 治理
governance:
	status: Completed
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	engineer: E2
	target_sprint: 3
	completed_on: 2025-10-02
	risk: "NaN 傳遞到 UI"
	mitigation: "守門將 NaN 轉 null"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_sanitizer.py -q
		- pytest services/tests/eval/test_kpi_aggregator.py -q
	deliverables:
		- services/eval/aggregation/sanitizer.py
		- services/tests/eval/test_sanitizer.py
	dod:
		- NaN / inf 轉為 None 並記錄於單元測試
		- Sanitizer 與 KPIAggregator 整合避免髒值流入報告
		- 文件記載資料衛生策略
	risk: "服務結構不一致降低重用"
	mitigation: "模板 + 目錄結構測試"
	status: Completed
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	engineer: E2
	target_sprint: 3
	completed_on: 2025-10-02
	risk: "部分寫入損壞 KPI 檔"
	mitigation: "原子 rename 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_kpi_writer.py -q
	deliverables:
		- services/eval/kpi_writer.py
		- services/tests/eval/test_kpi_writer.py
	dod:
		- fsync + 原子 rename 流程於測試中通過
		- 無法寫入時會拋出錯誤並於測試覆蓋
		- 文檔補充寫入策略
	estimate: 2p
	completed_at: 2025-10-01
	status: Completed
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	engineer: E2
	target_sprint: 3
	completed_on: 2025-10-02
	risk: "聚合延遲不可見"
	mitigation: "耗時指標 + 測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_kpi_aggregator.py -q
		- pytest services/tests/eval/test_execution.py -q
	deliverables:
		- services/eval/aggregation_metrics.py
		- services/tests/eval/test_kpi_aggregator.py
	dod:
		- 聚合耗時直方圖與計數指標可在 Prometheus 抓取
		- execute_evaluation_run 驗證指標輸出與檔案一致
		- README 補充聚合指標說明
		- 服務層級環境變數覆寫驗證並僅啟動時記錄一次

# TASK-003 治理
governance:
	status: Completed
	engineer: E3  # 前端需消費日誌格式; 若需可由 E1 實作
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-29
	risk: "非結構化日誌降低事故調查效率"
	mitigation: "JSON logger + trace_id middleware 測試 + 結構 schema"
	adr_impact: ["ADR-002"]
	ci_gate: ["unit-tests","log-schema-check"]
	verification:
		- pytest services/tests/test_logging.py services/tests/test_errors.py -q
	dod:
		- 日誌 schema 單元測試
		- 請求日誌含 trace id 測試
		- 錯誤路徑日誌含 trace id

# TASK-004 治理
governance:
	status: Completed
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-29
	risk: "錯誤格式不一致破壞 UI 處理"
	mitigation: "統一 handler + contract 測試樣板"
	adr_impact: ["ADR-003"]
	ci_gate: ["unit-tests","api-schema"]
	verification:
		- pytest services/tests/test_logging.py services/tests/test_errors.py -q
	dod:
		- 4xx/5xx 錯誤 schema snapshot
		- Trace id 一律存在
		- UI 錯誤解析測試通過

# TASK-005 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P2
	estimate: 2p
	risk: "物件儲存不穩造成資料遺失"
	mitigation: "重試 + checksum 失敗測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/common/test_object_store.py
	dod:
		- 重試邏輯測試
		- Checksum 不符失敗
		- README storage 區段
```

### 5.2 Ingestion 與 Processing
| ID       | Title                   | Description                          | Acceptance Criteria                                                       | Dependencies      | Artifacts                 | Req Mapping           |
|----------|-------------------------|--------------------------------------|---------------------------------------------------------------------------|-------------------|---------------------------|-----------------------|
| TASK-010 | Ingestion API           | POST /documents 受理 km_id/version   | 202 + job row                                                             | TASK-001          | services/ingestion/main.py、services/ingestion/repository.py、services/ingestion/openapi.json | FR ingest             |
| TASK-011 | KM 抓取與 Checksum      | 串流下載、計算 checksum、去重、儲存 raw | 重複返回既有 doc_id；僅一份 raw                                            | TASK-010,TASK-005 | ingestion/worker.py       | FR ingest             |
| TASK-012 | Ingestion 事件          | 發佈 document.ingested               | 測試抓到事件                                                              | TASK-011          | events/schema.py          | FR ingest 追溯        |
| TASK-013 | Processing Job API      | POST /process-jobs 啟動處理          | 202 job_id；驗證文件存在                                                   | TASK-011          | processing/api.py         | FR processing         |
| TASK-014 | 抽取/正規化階段         | PDF→text、unicode/空白清理            | 範例 PDF golden 通過                                                      | TASK-013          | processing/extract.py     | FR processing         |
| TASK-015 | 切片 + Embedding        | token 切片 + 批次 embedding          | chunks.jsonl 與向量數一致；單一 50k tokens 文件處理 p95 <30s（基線工作負載） | TASK-014          | processing/chunk_embed.py | FR processing         |
| TASK-016 | Processing 完成事件     | 發佈 document.processed              | 事件 schema 測試通過                                                      | TASK-015          | events/schema.py          | FR processing 追溯    |
| TASK-017 | UI Documents/Processing | Portal 表格 + 10s 輪詢               | 顯示文件與工作狀態                                                        | TASK-013          | insights-portal/...       | UI-FR-003/004,008~012 |

```yaml
# TASK-010 治理補充
governance:
	status: Done
	engineer: E1
	target_sprint: 1
# TASK-011 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion -q
	dod:
		- KM client 串流文件並依 km/version 與 checksum 去重
		- 物件儲存僅針對唯一內容上傳一次
		- Repository schema 新增工作狀態與錯誤欄位
# TASK-012 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion/test_worker.py
	dod:
		- 新文件完成後發佈 document.ingested 事件包
		- 重複匯入走 dedupe，不會觸發額外事件
		- 事件發布器支援注入以便於單元測試
# TASK-013 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-foundation@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/processing/test_jobs_api.py
	dod:
		- POST /process-jobs 成功返回 202 並持久化處理工作
		- document_id 不存在時回傳 document_not_found 錯誤封包 (404)
		- 無效請求載荷觸發統一驗證錯誤處理流程
# TASK-014 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-26
	verification:
		- pytest tests/services/processing/test_extraction_stage.py
	dod:
		- PDF 與純文字抽取共用正規化流程（whitespace + Unicode 清理）
		- 非文字二進位檔回傳 standardized unsupported_mime_type 錯誤
		- 空內容觸發 extraction_empty_text 錯誤保護下游
# TASK-016 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 3p
	risk: "缺少完成事件會破壞追蹤"
	mitigation: "Worker 編排器單元測試 + schema 驗證"
	adr_impact: ["ADR-004","ADR-006"]
	ci_gate: ["unit-tests","event-contract"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_worker.py -q
	deliverables:
		- services/processing/worker.py
		- tests/services/processing/test_worker.py
		- services/common/events.py
		- events/schemas/document.processed.v1.json
	dod:
		- Worker 串聯 抽取→切片→嵌入→持久化 並更新工作狀態
		- document.processed 事件包含 chunk_count、embedding_count、manifest_object_key、duration_ms
		- Schema 夾具同步 EN/ZH，README 新增追溯與 manifest 說明
# TASK-017 治理
governance:
	status: Done
	engineer: E3
	target_sprint: 1
	owner: ui-platform@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-30
	verification:
		- vitest run src/app/lifecycle/__tests__/DocumentsPanel.test.tsx src/app/lifecycle/__tests__/ProcessingPanel.test.tsx
	dod:
		- Documents 面板透過 10 秒輪詢顯示最新文件狀態、時間戳與失敗訊息
		- Processing 面板將所有處理工作以狀態徽章呈現，並支援手動重新整理
		- 輪詢 hook 整合 lifecycle store 更新並遵守 AbortController/timeout 安全機制
		- i18n 字串提供 zh-TW/en-US 的 lifecycle 分頁與空狀態文案
# TASK-015 治理
governance:
	status: Done          # Planned | In-Progress | Blocked | Done | Verified
	engineer: E1
	target_sprint: 1
	owner: platform-ml@team  # 負責角色 / 群組
	priority: P0             # P0 影響主流程
	estimate: 5p             # Story Points (相對複雜度)
	risk: "批次嵌入併發造成 GPU / RAM OOM 或速率限流"
	mitigation: "批次上限 512，失敗退避指數 3 次；環境變數 MAX_EMB_BATCH 可調"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["build-governance:schemas","perf-baseline"]
	slo:
		embedding_stage_p95_seconds: 30    # 單 document 處理 p95 目標
		embedding_error_rate_percent: 1.0  # <1% 失敗 (可重試後仍失敗)
	metrics:
		- processing_embedding_batch_duration_seconds
		- processing_embedding_batch_size
		- processing_embedding_error_total
	logs:
		- code=EMBED_BATCH_START level=INFO
		- code=EMBED_BATCH_FAIL  level=ERROR
	completed_at: 2025-09-30
	verification:
		- python3 -m pytest tests/processing -q
		- python3 -m pytest -q
	dod:                                   # Definition of Done checklist
		- 單元測試涵蓋 tokenizer / batch executor 錯誤路徑
		- 產生 chunks.jsonl 並校驗 chunk_count == embedding_count
		- 超過批次上限觸發降批訊息記錄
		- 指標成功出現在 /metrics
		- README 嵌入章節更新
		- 失敗事件回傳標準錯誤 envelope
```

#### TASK-015 子任務分解
| 子ID      | 標題             | 說明                                              | 驗收條件                                       | 依賴      | 產出                                | 備註              |
|-----------|------------------|---------------------------------------------------|------------------------------------------------|-----------|-------------------------------------|-------------------|
| TASK-015a | Tokenizer & 邊界 | 語言感知 tokenizer + 句/段落邊界，含 fallback      | 中英混合樣本邊界穩定；不支援 mime 觸發 fallback | TASK-015  | processing/stages/tokenizer.py      | 提供 tokens+spans |
| TASK-015b | Chunk 組裝規則   | 目標 512 tokens，overlap 可配置 (預設 50)          | 無 chunk 超過 800；相同 seed/config 決定性      | TASK-015a | processing/stages/chunk_rules.py    | 大小/重疊設定     |
| TASK-015c | 嵌入批次執行器   | 重試（指數退避）、每批 timeout、斷路器                | 429/timeout ≤3 重試；5 連續失敗開路並記錄       | TASK-015b | processing/stages/embed_executor.py | 指標+日誌         |
| TASK-015d | 持久化與完整性   | 寫入 chunks.jsonl + embeddings；計算 SHA256 & 清單 | 清單統計匹配；雜湊不符測試立即失敗              | TASK-015c | processing/stages/chunk_persist.py  | 下游輸入          |

```yaml
# TASK-015a 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	risk: "多語斷詞錯誤影響後續切片"
	mitigation: "混合語料 golden 測試 + fallback"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-26
	verification:
		- pytest tests/services/processing/test_tokenizer_stage.py
	dod:
		- 中英混合語料分句測試通過，輸出穩定
		- 不支援 mime 觸發降級日誌並回傳單一段落
		- 斷句策略記錄於 services/processing/README.md
# TASK-015b 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 1p
	risk: "chunk 過大導致延遲上升"
	mitigation: "硬上限 + 尺寸直方圖"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_chunk_rules.py -q
	deliverables:
		- services/processing/stages/chunk_rules.py
		- tests/services/processing/test_chunk_rules.py
	dod:
		- 無 >800 chunk
		- 決定性測試
		- 尺寸指標存在
		- 超過 hard_max 時調整 overlap 仍保留尾端順序
# TASK-015c 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P0
	estimate: 3p
	risk: "無界重試壓垮上游"
	mitigation: "最大次數 + 斷路測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-30
	verification:
		- pytest tests/services/processing/test_embed_executor.py -q
	deliverables:
		- services/processing/stages/embed_executor.py
		- tests/services/processing/test_embed_executor.py
		- services/common/config.py
		- services/pyproject.toml
	dod:
		- 重試/退避流程涵蓋 timeout 與 429 情境
		- 連續失敗後斷路器觸發並輸出指標
		- 統一 provider 例外為錯誤封包
		- Batch 大小受 settings.processing_embedding_max_batch_size 限制
		- Prometheus 指標輸出批次、延遲與失敗計數
# TASK-015d 治理
governance:
	status: Done
	engineer: E1
	target_sprint: 1
	owner: platform-processing@team
	priority: P1
	estimate: 2p
	risk: "完整性不符未被偵測"
	mitigation: "雜湊+計數 manifest 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-29
	verification:
		- pytest tests/services/processing/test_chunk_persistence.py -q
	deliverables:
		- services/processing/stages/chunk_persist.py
		- tests/services/processing/test_chunk_persistence.py
	dod:
		- Manifest schema 與 chunks.jsonl、embeddings.jsonl 一併輸出
		- 雜湊不符拋出 ChunkPersistenceError 並停止上傳
		- README 完整性章節描述 manifest + checksum 流程
```

### 5.3 Testset 生成
| ID       | Title                 | Description                         | Acceptance Criteria               | Dependencies | Artifacts           | Req Mapping               |
|----------|-----------------------|-------------------------------------|-----------------------------------|--------------|---------------------|---------------------------|
| TASK-020 | Testset Job API       | POST /testset-jobs + config hash    | 相同 config hash 一致             | TASK-016     | testset/api.py      | FR-013~016                |
| TASK-021 | 問答合成引擎          | 使用 RAGAS 函式 + seed              | 同 seed 產出一致前 N              | TASK-020     | testset/engine.py   | FR-013~016                |
| TASK-022 | Persona/Scenario 生成 | 產生 personas.json / scenarios.json | persona_count/scenario_count 正確 | TASK-021     | testset/persona.py  | FR-013~016, UI-FR-019~022 |
| TASK-023 | 去重與上限            | MinHash/集合去重 + 上限前截         | 重複率低於閾值                    | TASK-022     | testset/dedupe.py   | FR-013~016                |
| TASK-024 | Testset 事件          | 發佈 testset.created                | 事件驗證通過                      | TASK-023     | events/schema.py    | 追溯                      |
| TASK-025 | UI Testsets Panel     | 顯示 sample_count、seed、config_hash  | 正確輪詢更新                      | TASK-024     | insights-portal/... | UI-FR-019~022             |
#### TASK-020 子任務
| 子ID      | 標題                | 說明                                    | 驗收條件                                   | 依賴      | 產出                         | 備註           |
|-----------|---------------------|-----------------------------------------|--------------------------------------------|-----------|------------------------------|----------------|
| TASK-020a | 設定正規化與雜湊    | 正規化設定字段順序與預設後計算雜湊      | 相同語義設定得到相同 hash；空可選欄位忽略   | TASK-020  | testset/config_normalizer.py | 決定性基礎     |
| TASK-020b | Idempotent 工作守門 | 阻止相同 config+version 重複啟動        | 第二次送出 50ms 內回既有 job_id            | TASK-020a | testset/job_guard.py         | 利用 hash 索引 |
| TASK-020c | 驗證層              | Schema + 值域檢查（count, seed）          | 非法欄位 400 並含 error_code               | TASK-020a | testset/validation.py        | 可重用         |
| TASK-020d | 稽核日誌與指標      | 發佈建立 metric + 結構化日誌(hash,size) | testset_job_created_total 增加；日誌含 hash | TASK-020b | testset/metrics.py           | 可觀測         |

```yaml
# TASK-021 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-30
	risk: "seed 不穩定破壞重現"
	mitigation: "生成核心與引擎皆以固定 seed 單元測試驗證"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/testset/test_engine.py -q
		- pytest services/tests/testset/test_generator_core.py -q
	deliverables:
		- services/testset/engine.py
		- services/testset/repository.py
		- services/testset/generator_core.py
		- services/testset/persona_injector.py
		- services/testset/scenario_variation.py
		- services/testset/pre_filter.py
		- services/testset/payloads.py
		- services/tests/testset/test_engine.py
		- services/tests/testset/
	dod:
		- 引擎以決定性 object key/checksum 上傳樣本與 metadata
		- Repository 提供 running/completed 狀態更新並重置錯誤欄位
		- metadata.json 揭露 persona/scenario 計數、seed 與 checksum 以利追溯
		- 單元測試涵蓋成功路徑、缺漏工作、生成結果為空的錯誤處理
# TASK-022 治理
governance:
	status: Verified
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_engine.py -q
		- pytest services/tests/testset/test_persona_artifacts.py -q
	deliverables:
		- services/testset/persona.py
		- services/testset/engine.py
		- services/tests/testset/test_engine.py
		- services/tests/testset/test_persona_artifacts.py
	dod:
		- personas.json 與 scenarios.json 以決定性 schema 上傳並於 metadata.json 記錄對應路徑
		- Repository 完成態統計以產出清單的 count 作為來源確保數值一致
		- 單元測試驗證 artifact 內容、計數以及 metadata 關聯關係
# TASK-023 治理
governance:
	status: Done
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_dedupe.py -q
		- pytest services/tests/testset/test_generator_core.py -q
		- pytest services/tests/testset/test_engine.py -q
	deliverables:
		- services/testset/dedupe.py
		- services/testset/generator_core.py
		- services/tests/testset/test_dedupe.py
		- services/tests/testset/test_generator_core.py
	dod:
		- MinHash 類型去重流程濾除高度相似問答並保留 persona/scenario 多樣性
		- 生成 metadata 揭露 duplicate_ratio、deduplicated_count 與門檻供觀測
		- 品質過濾計量納入去重減少值，確保最終樣本數遵守設定上限
# TASK-024 治理
governance:
	status: Done
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_engine.py -q
	deliverables:
		- services/common/events.py
		- services/testset/engine.py
		- services/tests/testset/test_engine.py
	dod:
		- 成功生成後發佈 testset.created 事件，payload 含 testset_id、sample_count、seed、config_hash
		- EventPublisher 新增共用 helper 供服務重用
		- 單元測試以 JSON schema 驗證事件封包以確保契約穩定
# TASK-025 治理
governance:
	status: Done
	engineer: E3
	target_sprint: 2
	owner: ui-platform@team
	priority: P1
	estimate: 3p
	completed_at: 2025-09-30
	verification:
		- npm run test -- --run src/app/lifecycle/__tests__/TestsetsPanel.test.tsx
	deliverables:
		- insights-portal/src/app/lifecycle/TestsetsPanel.tsx
		- insights-portal/src/app/lifecycle/api.ts
		- insights-portal/src/app/lifecycle/types.ts
		- insights-portal/src/app/lifecycle/config.ts
		- insights-portal/src/app/store/usePortalStore.ts
		- insights-portal/src/app/i18n/index.ts
		- insights-portal/src/app/routes/LifecycleRoutes.tsx
		- insights-portal/src/app/lifecycle/__tests__/TestsetsPanel.test.tsx
	dod:
		- Testsets 面板以本地化欄位呈現 method、config hash、seed 與樣本/人格/情境計數並附狀態徽章
		- 輪詢 hook 支援手動刷新，並顯示 duplicate 標籤與錯誤訊息
		- i18n 資源與設定新增 testset 服務 base URL，確保 zh-TW / en-US 文案同步
# TASK-020a 治理
governance:
	status: Done
	engineer: E2
	target_sprint: 1
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "欄位順序導致 hash 漂移"
	mitigation: "正規化 golden 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- python3 -m pytest services/tests/testset/test_config_normalizer.py
	deliverables:
		- services/testset/config_normalizer.py
		- services/tests/testset/test_config_normalizer.py
	dod:
		- 順序不敏感測試
		- 空 optional 移除
		- Hash 文件
# TASK-020b 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "重複 job 浪費資源"
	mitigation: "執行緒競態測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py::test_duplicate_submission_returns_same_job_id -q
	dod:
		- 守門機制以 config_hash 重複時回傳既有 job_id
		- 底層 SQLite UNIQUE 限制保護資料一致性
		- 重複路徑輸出結構化除錯日誌（含 job_id 與狀態）
# TASK-020c 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 1
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	completed_at: 2025-09-30
	risk: "非法 count 造成爆量生成"
	mitigation: "範圍驗證 + 負面測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- python3 -m pytest services/tests/testset/test_validation.py -q
	deliverables:
		- services/testset/validation.py
		- services/tests/testset/test_validation.py
	dod:
		- 驗證模組拒絕缺漏欄位並回傳 error_code testset_config_invalid
		- seed 與樣本邊界皆有單元測試覆蓋
		- 策略列表去重並正規化 persona 資料
		- 範圍外請求回傳 400 並附細節
# TASK-020d 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "缺少稽核降低追溯"
	mitigation: "Metric + 日誌整合測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py -q
	dod:
		- Prometheus counter 以 created/duplicate label 追蹤提交結果
		- 結構化日誌輸出 job_id、config_hash、method 方便追蹤
		- /metrics 端點公開 testset_job_created_total 供監控抓取
```

#### TASK-021 子任務
| 子ID      | 標題              | 說明                           | 驗收條件                              | 依賴      | 產出                          | 備註     |
|-----------|-------------------|--------------------------------|---------------------------------------|-----------|-------------------------------|----------|
| TASK-021a | 種子生成核心      | Q/A 生成主迴圈與 seed 控制     | 固定 seed 前 5 組 Q/A 一致            | TASK-021  | testset/generator_core.py     | 決定性   |
| TASK-021b | Persona 注入層    | 對 prompt 注入 persona context | persona token 加入；snapshot diff 通過 | TASK-021a | testset/persona_injector.py   | 可擴充   |
| TASK-021c | Scenario 變異模組 | 依 scenario 規則變化 context   | 每文件組 ≥3 個 scenario 變體          | TASK-021b | testset/scenario_variation.py | 多樣性   |
| TASK-021d | 品質與初步去重    | 輕量模糊比對 + 長度界限 預過濾 | 去除 ≥90% trivial duplicates          | TASK-021c | testset/pre_filter.py         | 效能前置 |

```yaml
# TASK-021a 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-30
	risk: "seed 不穩定破壞重現"
	mitigation: "固定 seed 單元測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/testset/test_generator_core.py -q
	deliverables:
		- services/testset/generator_core.py
		- services/testset/payloads.py
		- services/tests/testset/test_generator_core.py
	dod:
		- 前 5 組穩定
		- seed 參數文件
		- Snapshot 認可
# TASK-021b 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "persona 注入令 prompt 過長"
	mitigation: "token 長度閾值測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_persona_injector.py -q
	deliverables:
		- services/testset/persona_injector.py
		- services/tests/testset/test_persona_injector.py
	dod:
		- token overhead < limit
		- snapshot 更新
		- README persona 章節
# TASK-021c 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P2
	estimate: 1p
	risk: "變異不足"
	mitigation: "變體數量斷言"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_scenario_variation.py -q
	deliverables:
		- services/testset/scenario_variation.py
		- services/tests/testset/test_scenario_variation.py
	dod:
		- ≥3 變體測試
		- 多樣性說明
		- Scenario 規則文件
# TASK-021d 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "低品質或重複未濾除"
	mitigation: "模糊相似度閾值測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_pre_filter.py -q
	deliverables:
		- services/testset/pre_filter.py
		- services/tests/testset/test_pre_filter.py
	dod:
		- 重複移除 >=90%
		- 長度邊界測試
		- pre-filter 指標
```

```yaml
# TASK-010 治理
governance:
	status: Done
	engineer: E1
	owner: platform-ingestion@team
	priority: P1
	estimate: 2p
	risk: "缺少驗證允許無效文件進入管線"
	mitigation: "Pydantic schema + 負面測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-25
	verification:
		- pytest tests/services/ingestion/test_documents_api.py
	dod:
		- 202 + job_id 測試
		- 無效 payload 400
		- OpenAPI 已更新

# TASK-020 治理
governance:
	status: Completed
	engineer: E2
	owner: platform-testset@team
	priority: P1
	estimate: 1p
	risk: "設定雜湊不穩定破壞快取"
	mitigation: "正規化 golden 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/testset/test_api.py -q
		- pytest services/tests/testset/test_validation.py -q
	dod:
		- POST /testset-jobs 以 202 回傳 job_id、config_hash 與 method
		- 重複提交重用相同 job_id 並回傳 duplicate=true
		- Metrics 以 created/duplicate label 紀錄每次請求
		- 結構化日誌含 job_id 與 config_hash 便於追溯
```

### 5.4 Evaluation Runner
| ID       | Title              | Description                        | Acceptance Criteria         | Dependencies | Artifacts           | Req Mapping   |
|----------|--------------------|------------------------------------|-----------------------------|--------------|---------------------|---------------|
| TASK-030 | Evaluation Run API | 建立 run (testset_id+profile)      | 202 run_id                  | TASK-024     | services/eval/main.py | FR-017~022    |
| TASK-031 | RAG 呼叫與上下文   | RAG 介接與 contexts 收集           | evaluation_items 含 context | TASK-030     | eval/rag_adapter.py | FR-017~022    |
| TASK-032 | 指標外掛註冊       | 動態載入基礎指標                   | 指標全部執行                | TASK-031     | eval/metrics/*      | FR-017~022    |
| TASK-033 | 評估項目持久化     | 流式寫 evaluation_items.json       | 行數 == samples             | TASK-032     | services/eval/persistence_pipeline.py | FR-017~022    |
| TASK-034 | KPI 聚合           | 產出 kpis.json                     | p50/p95 正確                | TASK-033     | services/eval/aggregation/aggregator.py   | FR-017~022    |
| TASK-035 | run.completed 事件 | 發佈完成（counts + metrics_version） | 測試驗證                    | TASK-034     | events/schema.py    | 追溯          |
| TASK-036 | UI Evaluation Runs | 顯示進度、verdict、error_count       | 輪詢更新                    | TASK-035     | insights-portal/... | UI-FR-023~026 |

```yaml
# TASK-031 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 2p
	completed_at: 2025-09-30
	risk: "缺少 context 捕捉會讓後續指標與追蹤失真"
	mitigation: "EvaluationRunner 強制空上下文 fallback 並以單元測試覆蓋重試遙測"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["unit-tests"]
	verification:
		- python3 -m pytest services/tests/eval/test_runner.py
	deliverables:
		- services/eval/runner.py
		- services/tests/eval/test_runner.py
		- services/eval/retry_policy.py
	dod:
		- EvaluationRunner 透過 RetryPolicy 串接 RAG adapter 並輸出重試指標欄位
		- 空或失敗呼叫會注入標記 empty_context 的後援上下文
		- metadata 新增 rag_attempts、rag_outcome、rag_error_code 供後續持久化使用
# TASK-033 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	completed_on: 2025-10-02
	verification:
		- pytest services/tests/eval/test_stream_writer.py -q
		- pytest services/tests/eval/test_persistence_pipeline.py -q
		- pytest services/tests/eval/test_execution.py -q
	deliverables:
		- services/eval/persistence_pipeline.py
		- services/eval/stream_writer.py
		- services/eval/backpressure.py
		- services/eval/manifest.py
		- services/tests/eval/test_execution.py
	dod:
		- 串流寫入遵守 flush 週期並在測試中驗證行數遞增
		- 佇列工人提供有界背壓並於飽和時發出告警遙測
		- finalize 產出 evaluation_items.jsonl、manifest 與 kpis.json 並通過計數驗證
		- execute_evaluation_run 確保每個樣本的指標與檔案行數一致
# TASK-034 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	completed_on: 2025-10-02
	verification:
		- pytest services/tests/eval/test_distribution.py -q
		- pytest services/tests/eval/test_kpi_aggregator.py -q
		- pytest services/tests/eval/test_execution.py -q
	deliverables:
		- services/eval/aggregation/aggregator.py
		- services/eval/aggregation/distribution.py
		- services/eval/aggregation/sanitizer.py
		- services/eval/kpi_writer.py
		- services/eval/aggregation_metrics.py
	dod:
		- 聚合計算出 min/max/平均/p50/p95 並處理 NaN 值
		- KPI 寫入採原子 rename 並確保落盤時 fsync
		- Prometheus 指標針對聚合耗時與總筆數提供觀測
		- kpis.json 與計數資料在執行測試中一致
# TASK-035 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	completed_at: 2026-03-06
	deliverables:
		- services/common/events.py (run_completed + report_completed)
# TASK-036 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 3
	completed_at: 2025-09-30
# TASK-030 治理
governance:
	status: Completed
	owner: platform-eval@team
	priority: P1
	engineer: E2
	target_sprint: 2
	completed_at: 2025-10-02
	estimate: 2p
	risk: "執行狀態生命週期不清晰"
	mitigation: "狀態圖 + 轉移測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_api.py -q
		- pytest services/tests/eval/test_validation.py -q
	deliverables:
		- services/eval/main.py
		- services/eval/repository.py
		- services/eval/schemas.py
		- services/tests/eval/test_api.py
		- services/tests/eval/test_validation.py
	dod:
		- POST /eval-runs 回傳 202 並持久化 run 資料
		- 重複提交 reused run_id 並透過 guard 阻擋重建
		- /metrics 公開 evaluation_run_created_total 計數
		- 驗證層對不存在的 testset_id 回傳結構化錯誤
		- 治理文件同步更新
```

#### TASK-030 子任務
| 子ID      | 標題         | 說明                                 | 驗收條件                                     | 依賴      | 產出               | 備註     |
|-----------|--------------|--------------------------------------|----------------------------------------------|-----------|--------------------|----------|
| TASK-030a | Run 狀態模型 | 定義執行狀態 enum + 狀態轉移         | 非法轉移被拒；狀態圖測試通過                  | TASK-030  | eval/run_states.py | 治理基礎 |
| TASK-030b | 輸入驗證層   | 驗證 testset_id 與 profile 存在      | 非法 profile 回 400 含 error_code            | TASK-030a | eval/validation.py | 可重用   |
| TASK-030c | 冪等送出守門 | 阻止相同 testset+profile 重複執行    | 第二次請求回相同 run_id                      | TASK-030b | eval/run_guard.py  | 決定性   |
| TASK-030d | 指標與日誌   | 發佈 run_created metric + 結構化日誌 | run_created_total 增加；日誌含 run_id/profile | TASK-030c | eval/metrics.py    | 可觀測   |

```yaml
# TASK-030a 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_run_states.py -v
	deliverables:
		- services/eval/run_states.py
		- services/tests/eval/test_run_states.py
	estimate: 1p
	risk: "狀態轉移錯誤造成非預期流程"
	mitigation: "有限狀態表 + 轉移單元測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 狀態轉移測試完成 (26 個測試通過)
		- 非法轉移 raise InvalidStateTransitionError
		- RunState enum 含 5 個狀態 (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
		- 終端狀態偵測實作完成
		- 狀態轉移驗證與完整測試覆蓋率
# TASK-030b 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	completed_at: 2025-09-30
	estimate: 1p
	risk: "缺少驗證導致錯誤 run config"
	mitigation: "Pydantic schema + 負面測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_validation.py -v -k "not trio"
	deliverables:
		- services/eval/validation.py
		- services/tests/eval/test_validation.py
	dod:
		- 非法 profile 測試完成 (19 個測試通過)
		- 缺 testset_id 測試含 TestsetNotFoundError
		- 錯誤 envelope 文件與 ServiceError 整合
		- EvaluationRunCreateRequest schema 含 pydantic 驗證
		- testset_id UUID 格式驗證
		- profile 名稱驗證 (英數字 + 底線)
		- Timeout 與 retry 參數範圍驗證
		- EvaluationRunValidator 含非同步 testset 存在檢查
# TASK-030c 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P1
	completed_at: 2025-10-02
	estimate: 1p
	risk: "重複執行浪費資源"
	mitigation: "Hash 守門 + 併發測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_api.py -k "idempotent" -q
	deliverables:
		- services/eval/run_guard.py
		- services/tests/eval/test_api.py
	dod:
		- 重複請求取得相同 run_id（API 測試驗證 JSON 相等）
		- 守門器透過 repository 檢查阻擋同時存在的 active run
		- 結構化日誌包含 result 欄位標示 duplicate 情境
# TASK-030d 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 2
	owner: platform-eval@team
	priority: P2
	completed_at: 2025-10-02
	estimate: 1p
	risk: "缺少指標降低可觀測"
	mitigation: "指標命名 lint + 抓取測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","build-governance:schemas"]
	verification:
		- pytest services/tests/eval/test_api.py -k "metrics" -q
	deliverables:
		- services/eval/metrics.py
		- services/eval/main.py
		- services/tests/eval/test_api.py
	dod:
		- /metrics 端點公開 evaluation_run_created_total 計數器
		- Run 建立時發出含 run_id 與 profile 的結構化日誌
		- API 整合測試覆蓋 Prometheus registry 接線
		- Metric 暴露
		- 結構化日誌測試
		- 文件更新
```

#### TASK-031 子任務
| 子ID      | 標題             | 說明                        | 驗收條件                                   | 依賴      | 產出                    | 備註       |
|-----------|------------------|-----------------------------|--------------------------------------------|-----------|-------------------------|------------|
| TASK-031a | Adapter 介面     | 定義呼叫 RAG 系統介面       | 缺 method 拋 NotImplemented                | TASK-031  | eval/rag_interface.py   | 契約       |
| TASK-031b | Context 捕捉包裝 | 包裝呼叫並儲存檢索 contexts | 每個 evaluation item context 陣列 length>0 | TASK-031a | eval/context_capture.py | 可觀測     |
| TASK-031c | 重試與逾時策略   | 實作重試＋抖動＋逾時          | 429/timeout ≤3 次；最終錯誤記錄             | TASK-031b | eval/retry_policy.py    | 韌性       |
| TASK-031d | 指標與 Trace ID  | 發佈延遲直方圖 + trace id   | rag_request_latency_seconds 存在           | TASK-031c | eval/metrics.py         | 效能能見度 |

```yaml
# TASK-031a 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "介面漂移破壞 adapter"
	mitigation: "抽象基類 + 契約測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_rag_interface.py -q
	deliverables:
		- services/eval/rag_interface.py
		- services/tests/eval/test_rag_interface.py
	dod:
		- 基底 adapter 呼叫 invoke 時 raise NotImplementedError 測試通過
		- 契約註解描述 request/response 結構
		- 提供 StaticResponseAdapter 供單元測試引用
# TASK-031b 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "未捕捉 context 影響評估品質"
	mitigation: "Wrapper 測試 contexts length>0"
	adr_impact: []
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_context_capture.py -q
	deliverables:
		- services/eval/context_capture.py
		- services/tests/eval/test_context_capture.py
	dod:
		- 正常情境保留 RAG contexts 並新增 metadata
		- 空檢索自動注入 fallback context（單元測試覆蓋）
		- 補充文件描述 context capture 行為
# TASK-031c 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P0
	estimate: 2p
	risk: "無界重試造成延遲"
	mitigation: "最大次數 + 抖動退避測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_retry_policy.py -q
	deliverables:
		- services/eval/retry_policy.py
		- services/tests/eval/test_retry_policy.py
	dod:
		- 429 與 Timeout 案例重試次數 ≤3（測試驗證）
		- TimeoutError 處理流程通過單元測試
		- Telemetry 物件提供後續指標串接
# TASK-031d 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "缺少延遲可視性"
	mitigation: "直方圖 + 抓取測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_metrics.py -q
	deliverables:
		- services/eval/metrics.py
		- services/tests/eval/test_metrics.py
		- services/eval/README.md
	dod:
		- rag_request latency/attempt 指標公開（單元測試驗證）
		- 結構化日誌包含 trace_id 與錯誤碼
		- README 更新記錄新增指標
```

#### TASK-033 子任務
| 子ID      | 標題            | 說明                                         | 驗收條件                           | 依賴      | 產出                  | 備註   |
|-----------|-----------------|----------------------------------------------|------------------------------------|-----------|-----------------------|--------|
| TASK-033a | 串流寫入核心    | 追加寫入 +  flush 週期                       | 檔案行數遞增；flush 週期 ≤ 設定秒數 | TASK-033  | services/eval/stream_writer.py | 效率   |
| TASK-033b | 背壓處理        | 佇列上限 + 丟棄策略                          | 模擬慢磁碟觸發背壓日誌             | TASK-033a | services/eval/backpressure.py  | 穩定   |
| TASK-033c | 完整性 Manifest | 維護計數 + checksum                          | 最終計數匹配；異常測試失敗          | TASK-033b | services/eval/manifest.py      | 追溯   |
| TASK-033d | 指標輸出        | items_written counter + flush latency 直方圖 | 指標在 /metrics 可見               | TASK-033c | services/eval/persistence_metrics.py       | 可觀測 |

```yaml
# TASK-033a 治理
governance:
	status: Completed
	engineer: E2 (E1 協助 I/O)
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	completed_on: 2025-10-02
	risk: "Flush 延遲導致崩潰資料遺失"
	mitigation: "Flush 間隔測試 + fsync 選項"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_stream_writer.py -q
	deliverables:
		- services/eval/stream_writer.py
		- services/tests/eval/test_stream_writer.py
	dod:
		- Flush 間隔 (含 0) 於單元測試中驗證確實落盤
		- 緩衝寫入計算 bytes/items 並同步指標
		- JSONL 寫入同步產出 manifest 資訊
# TASK-033b 治理
governance:
	status: Completed
	engineer: E2 (E1 協助)
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	completed_on: 2025-10-02
	risk: "背壓未處理導致 OOM"
	mitigation: "有限隊列 + 丟棄指標"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_backpressure.py -q
	deliverables:
		- services/eval/backpressure.py
		- services/tests/eval/test_backpressure.py
	dod:
		- 佇列容量受限並在飽和時釋出警告與 drop 指標
		- 慢速寫入模擬測試驗證無死結且最終排空
		- 丟棄次數被計入遙測供觀測性使用
# TASK-033c 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	completed_on: 2025-10-02
	risk: "計數不一致未被偵測"
	mitigation: "Manifest 驗證測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_manifest.py -q
	deliverables:
		- services/eval/manifest.py
		- services/tests/eval/test_manifest.py
	dod:
		- Manifest 紀錄 bytes/count 並提供 mismatch 例外
		- 失敗案例在單元測試中觸發 RuntimeError
		- Docstring 補充格式與欄位定義
# TASK-033d 治理
governance:
	status: Completed
	engineer: E2
	target_sprint: 3
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	completed_on: 2025-10-02
	risk: "缺少指標降低營運能見度"
	mitigation: "指標命名 lint"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	verification:
		- pytest services/tests/eval/test_persistence_metrics.py -q
		- pytest services/tests/eval/test_execution.py -q
	deliverables:
		- services/eval/persistence_metrics.py
		- services/tests/eval/test_persistence_metrics.py
	dod:
		- items_written counter 與 flush latency 直方圖成功曝光
		- 指標在執行流程中被驗證並可於 Prometheus 抓取
		- README 補充持久化指標用途
```

#### TASK-034 子任務
| 子ID      | 標題           | 說明                       | 驗收條件                                              | 依賴      | 產出                      | 備註     |
|-----------|----------------|----------------------------|-------------------------------------------------------|-----------|---------------------------|----------|
| TASK-034a | 分佈計算器     | 計算 p50, p95, min, max    | 測試固定輸入得期望值                                  | TASK-034  | services/eval/aggregation/distribution.py      | 核心邏輯 |
| TASK-034b | 聚合完整性守門 | 驗證指標 schema + NaN 轉換 | NaN → null；schema 驗證通過                            | TASK-034a | services/eval/aggregation/sanitizer.py | 資料衛生 |
| TASK-034c | KPI 寫檔器     | 原子寫入 kpis.json         | 臨時檔 rename；部分寫入測試被阻止                      | TASK-034b | services/eval/kpi_writer.py        | 可靠性   |
| TASK-034d | 指標發布       | 聚合耗時 + 總筆數指標      | aggregation_duration_seconds & kpi_records_total 存在 | TASK-034c | services/eval/aggregation_metrics.py           | 可觀測   |

```yaml
# TASK-034a 治理
governance:
	status: Planned
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "分位數計算錯誤"
	mitigation: "固定樣本測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 分位數測試
		- 負值處理
		- 文件更新
	engineer: E2
	target_sprint: 3
# TASK-034b 治理
governance:
	status: Planned
	owner: platform-eval@team
	priority: P1
	estimate: 1p
	risk: "NaN 傳遞到 UI"
	mitigation: "守門將 NaN 轉 null"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- NaN 守門測試
		- Schema 驗證
		- README 衛生說明
	engineer: E2
	target_sprint: 3
# TASK-034c 治理
governance:
	status: Planned
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "部分寫入損壞 KPI 檔"
	mitigation: "原子 rename 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 原子寫入測試
		- 暫存檔清理
		- 文件更新
	engineer: E2
	target_sprint: 3
# TASK-034d 治理
governance:
	status: Planned
	owner: platform-eval@team
	priority: P2
	estimate: 1p
	risk: "聚合延遲不可見"
	mitigation: "Duration 指標 + 測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- Duration 指標
		- Records counter
		- README metrics 更新
	engineer: E2
	target_sprint: 3
```

```yaml
# TASK-032 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-observability@team
	priority: P1
	estimate: 3p
	risk: "外掛初始化異常導致整個評估流程中斷"
	mitigation: "隔離匯入 (importlib.reload) + 單外掛失敗轉降級並記錄 metrics"
	adr_impact: ["ADR-001","ADR-005"]
	ci_gate: ["unit-tests","lint","build-governance:schemas"]
	plugin_contract_version: 1
	failure_isolation: try-except per-metric with fallback noop
	slo:
		registry_init_seconds_p95: 1.5
		plugin_failure_rate_percent: 0.5
	metrics:
		- eval_metrics_registry_load_seconds
		- eval_metric_execution_duration_seconds
		- eval_metric_failure_total
	logs:
		- code=METRIC_PLUGIN_REGISTERED level=INFO
		- code=METRIC_PLUGIN_FAILED     level=ERROR
	dod:
		- 至少 3 個基礎指標 (faithfulness, answer_relevancy, precision) 可載入
		- 單外掛故意拋出例外不影響其它指標執行
		- registry 列表具 deterministic 排序 (名稱字母序)
		- /metrics 暴露載入耗時與失敗計數
		- README metrics 章節新增外掛接點說明
	deliverables:
		- services/eval/metrics/__init__.py
		- services/eval/metrics/interface.py
		- services/eval/metrics/loader.py
		- services/eval/metrics/baseline/
```

#### TASK-032 子任務分解
| 子ID      | 標題            | 說明                                        | 驗收條件                                           | 依賴      | 產出                       | 備註     |
|-----------|-----------------|---------------------------------------------|----------------------------------------------------|-----------|----------------------------|----------|
| TASK-032a | 載入契約        | 定義 MetricPlugin 介面與版本常數            | 缺 method 拋清楚錯誤；版本經 registry endpoint 暴露 | TASK-032  | eval/metrics/interface.py  | 介面固定 |
| TASK-032b | 基礎指標實作    | faithfulness / answer_relevancy / precision | 測試皆回數值；固定 seed 決定性                      | TASK-032a | eval/metrics/baseline/*.py | 核心指標 |
| TASK-032c | 探索 & 失敗隔離 | 檔案/entrypoints 探索 + try/except 隔離     | 故意壞外掛被跳過且計數+日誌                        | TASK-032b | eval/metrics/loader.py     | 隔離驗證 |

```yaml
# TASK-032a 治理
governance:
	status: Completed
	owner: platform-observability@team
	priority: P1
	estimate: 1p
	risk: "介面不清導致外掛失敗"
	mitigation: "契約測試 + 註解"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	started_at: 2025-09-30
	completed_at: 2025-09-30
	progress_notes:
		- 已在 services/eval/metrics/interface.py 建立 MetricInput/MetricValue 資料類別與 MetricPlugin 協定。
		- 新增 validate_plugin 輔助函式，缺少 evaluate 方法會拋出 MetricPluginDefinitionError。
		- 新測試 services/tests/eval/test_metric_interface.py 驗證契約版本常數與錯誤訊息。
	verification:
		- pytest services/tests/eval/test_metric_interface.py -q
	deliverables:
		- services/eval/metrics/interface.py
		- services/tests/eval/test_metric_interface.py
	dod:
		- 介面測試
		- 缺方法錯誤
		- 契約文件
# TASK-032b 治理
governance:
	status: Completed
	owner: platform-observability@team
	priority: P1
	estimate: 2p
	risk: "基礎指標分數不一致"
	mitigation: "固定樣本 + seed"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_baseline_metrics.py -q
	deliverables:
		- services/eval/metrics/baseline/__init__.py
		- services/eval/metrics/baseline/faithfulness.py
		- services/eval/metrics/baseline/answer_relevancy.py
		- services/eval/metrics/baseline/context_precision.py
		- services/tests/eval/test_baseline_metrics.py
	progress_notes:
		- 建立三個字詞重疊基線指標（faithfulness、answer_relevancy、context_precision），並確保輸出決定性與四捨五入規則。
		- 新增共享 token 化工具，處理中英混合內容且不需額外相依套件。
		- 指標 metadata 輸出 matched/total 統計，協助後續 KPI 聚合。
	dod:
		- 3 指標實作
		- 決定性測試
		- README 指標列表
	README_updates:
		- services/eval/README.md#baseline-metric-plugins
# TASK-032c 治理
governance:
	status: Completed
	owner: platform-observability@team
	priority: P1
	estimate: 2p
	risk: "壞外掛使 registry 崩潰"
	mitigation: "隔離 try/except 測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	completed_at: 2025-09-30
	verification:
		- pytest services/tests/eval/test_metrics_loader.py -q
		- pytest services/tests/eval -q
	deliverables:
		- services/eval/metrics/loader.py
		- services/tests/eval/test_metrics_loader.py
	progress_notes:
		- 完成 metric registry，支援內建、字串路徑匯入與 entry point 探索，並以名稱排序去重。
		- 透過 try/except 隔離外掛失敗，METRIC_PLUGIN_FAILED 日誌與失敗計數同步累計，並紀錄每個外掛執行耗時直方圖。
		- 新增測試涵蓋內建註冊、插件路徑載入、entry point 模擬及故障隔離情境。
	dod:
		- 故障注入測試
		- 跳過警告日誌
		- 失敗計數指標
```

### 5.5 Insights Adapter 與 Reporting
| ID       | Title                 | Description                                | Acceptance Criteria        | Dependencies      | Artifacts                  | Req Mapping               |
|----------|-----------------------|--------------------------------------------|----------------------------|-------------------|----------------------------|---------------------------|
| TASK-040 | Insights 正規化       | 轉換 evaluation artifacts 為 portal schema | export_summary.json (旗標) | TASK-035          | adapter/normalize.py       | FR-038/039                |
| TASK-041 | 報告 HTML 模板        | 行政 + 技術頁面                            | HTML 渲染一致              | TASK-040          | reporting/templates/*.html | FR-037~040                |
| TASK-042 | PDF 生成              | Playwright 轉 PDF + run_meta 更新          | pdf_url 可用               | TASK-041          | reporting/pdf.py           | FR-037~040                |
| TASK-043 | report.completed 事件 | 成功 PDF/HTML 後發佈                       | 驗證通過                   | TASK-042          | events/schema.py           | FR-037~040                |
| TASK-044 | UI Reports Panel      | 顯示報告與 fallback                        | PDF 404 → HTML             | TASK-043          | insights-portal/...        | UI-FR-030~032             |
| TASK-045 | KM 摘要匯出           | 產出 testset/kg summary (counts)           | JSON schema 通過           | TASK-024,TASK-060 | adapter/km_export.py       | FR-041/042, UI-FR-033~035 |
| TASK-046 | UI KM Summaries       | 顯示摘要 + delta                           | 差異計算正確               | TASK-045          | insights-portal/...        | UI-FR-033~035             |

### 5.6 知識圖譜（旗標功能）
```yaml
# TASK-040 治理
governance:
	status: Completed
	owner: platform-reporting@team
	priority: P1
	estimate: 2p
	risk: "Pipeline 與 UI schema 漂移"
	mitigation: "Golden snapshot 測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- export_summary.json snapshot
		- 預設旗標關閉
		- README 使用說明
	engineer: E2
	target_sprint: 3
	completed_at: 2026-03-06
	deliverables:
		- services/adapter/normalize.py
		- services/tests/adapter/test_normalize.py
# TASK-041 治理
governance:
	status: Completed
	owner: platform-reporting@team
	priority: P1
	estimate: 2p
	risk: "模板變數不匹配"
	mitigation: "模板渲染矩陣測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 行政模板測試
		- 技術模板測試
		- 佔位符覆蓋 100%
	engineer: E2
	target_sprint: 3
	completed_at: 2026-03-06
	deliverables:
		- services/reporting/templates/executive.html
		- services/reporting/templates/technical.html
# TASK-042 治理
governance:
	status: Completed
	owner: platform-reporting@team
	priority: P1
	estimate: 1p
	risk: "PDF 渲染不穩定"
	mitigation: "固定 viewport + 字型嵌入"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- PDF 大小 >0 測試
		- run_meta 更新
		- fallback 邏輯文件
	engineer: E2
	target_sprint: 3
	completed_at: 2026-03-06
	deliverables:
		- services/reporting/pdf.py
		- services/tests/adapter/test_pdf.py
# TASK-043 治理
governance:
	status: Completed
	owner: platform-reporting@team
	priority: P2
	estimate: 1p
	risk: "事件早於持久化送出"
	mitigation: "順序測試確保後送"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- 事件 schema 測試
		- 發佈順序測試
		- README 事件說明
	engineer: E2
	target_sprint: 3
	completed_at: 2026-03-06
	notes: "report_completed() 與 run_completed() 同一 PR 實作"
# TASK-044 治理
governance:
	status: Completed
	owner: platform-ui@team
	priority: P2
	estimate: 2p
	risk: "PDF 缺失 UI 處理不佳"
	mitigation: "Fallback 整合測試"
	adr_impact: []
	ci_gate: ["ui-tests"]
	dod:
		- Fallback 截圖測試
		- Loading skeleton 測試
		- 無障礙掃描
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- insights-portal/src/app/lifecycle/ReportsPanel.tsx
		- services/reporting/main.py (完整路由)
# TASK-045 治理
governance:
	status: Completed
	owner: platform-reporting@team
	priority: P2
	estimate: 1p
	risk: "摘要計數不正確"
	mitigation: "與原始 artifacts 交叉比對"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- testset_summary_v0 schema 測試
		- kg_summary_v0 schema 測試
		- README 摘要格式
	engineer: E2
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/adapter/km_export.py
		- services/tests/adapter/test_km_export.py
```
| ID       | Title                   | Description                          | Acceptance Criteria      | Dependencies | Artifacts           | Req Mapping           |
|----------|-------------------------|--------------------------------------|--------------------------|--------------|---------------------|-----------------------|
| TASK-060 | KG Builder API          | 受理建構請求與狀態查詢               | 202 kg_id                | TASK-015     | kg/api.py           | UI-FR-016~018         |
| TASK-061 | 實體與關鍵詞抽取        | spaCy + KeyBERT 混合                 | 範例驗證出實體           | TASK-060     | kg/extract.py       | UI-FR-016~018         |
| TASK-062 | 關係建構整合            | Jaccard/Overlap/Cosine/SummaryCosine | relationships 非空       | TASK-061     | kg/relationships.py | UI-FR-016~018         |
| TASK-063 | KG 摘要 Endpoint        | counts + histogram + top_entities    | schema 驗證              | TASK-062     | kg/summary.py       | UI-FR-016~018         |
| TASK-064 | UI KG 摘要面板          | 顯示統計（旗標 true 時）               | 旗標 false 顯示 fallback | TASK-063     | insights-portal/... | UI-FR-016~018         |
| TASK-065 | Lazy 視覺化 (Cytoscape) | 動態載入，節點上限 500                | chunk 分離、渲染成功      | TASK-064     | insights-portal/... | UI-FR-018, UI-NFR-006 |
| TASK-066 | Subgraph API 草稿實作   | 決定性抽樣子圖                       | truncated flag 穩定      | TASK-063     | kg/subgraph.py      | Spec §27              |
| TASK-067 | UI Subgraph 聚焦        | 節點/實體聚焦子圖請求                | 顯示 sampling pill       | TASK-066     | insights-portal/src/app/lifecycle/KgPanel.tsx | ✅ 已完成         |

#### TASK-062 子任務分解
| 子ID      | 標題                 | 說明                                  | 驗收條件                               | 依賴      | 產出                         | 備註              |
|-----------|----------------------|---------------------------------------|----------------------------------------|-----------|------------------------------|-------------------|
| TASK-062a | 節點屬性增豐         | 實體/關鍵詞/摘要/embedding 填充       | 範例節點含必要屬性；缺 embedding 有標誌 | TASK-062  | kg/extract.py                | 相似度前置        |
| TASK-062b | Jaccard & Overlap    | Jaccard(entities)+Overlap(keyphrases) | 範例 CSV >0 relationships；閾值可調     | TASK-062a | kg/relationships.py          | 非 embedding 路徑 |
| TASK-062c | 向量 Cosine Fallback | 有 embedding 算 cosine；無則跳過       | 無 embedding 不報錯；記錄數量           | TASK-062b | kg/relationships.py          | 品質提升選項      |
| TASK-062d | 閾值調參工具         | 評估不同閾值關係數 & 平均相似         | 產出 JSON 報告含 count & 平均值        | TASK-062c | scripts/kg_threshold_tune.py | 指南              |

```yaml
# TASK-046 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- insights-portal/src/app/lifecycle/KmSummariesPanel.tsx
# TASK-060 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/kg/main.py (完整 FastAPI 路由)
		- services/kg/repository.py
		- services/tests/kg/test_repository.py
# TASK-061 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/kg/extract.py
		- services/tests/kg/test_extract.py
# TASK-062 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/kg/relationships.py
		- services/tests/kg/test_relationships.py
# TASK-063 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/kg/summary.py
		- services/tests/kg/test_summary.py
# TASK-064 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- insights-portal/src/app/lifecycle/KgPanel.tsx (feature-flagged)
# TASK-065 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	notes: "KgPanel.tsx feature-flagged；Cytoscape lazy import 延後至後續"
# TASK-066 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 4
	completed_at: 2026-03-06
	deliverables:
		- services/kg/subgraph.py
		- services/tests/kg/test_subgraph.py
# TASK-062a 治理
governance:
	status: Completed
	owner: platform-kg@team
	priority: P1
	estimate: 2p
	risk: "節點屬性缺失降低關係品質"
	mitigation: "屬性完整性測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 節點含 entities/keyphrases
		- 缺 embedding 標誌
		- README 屬性說明
	completed_at: 2026-03-06
	deliverables:
		- services/kg/extract.py
# TASK-062b 治理
governance:
	status: Completed
	owner: platform-kg@team
	priority: P1
	estimate: 1p
	risk: "閾值過嚴為 0 關係"
	mitigation: "參數掃描測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- >0 關係樣本
		- 閾值設定文件
		- 日誌含 count
	completed_at: 2026-03-06
	deliverables:
		- services/kg/relationships.py (build_jaccard_relationships, build_overlap_relationships)
# TASK-062c 治理
governance:
	status: Completed
	owner: platform-kg@team
	priority: P2
	estimate: 1p
	risk: "無 embedding 路徑報錯"
	mitigation: "skip 邏輯測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 優雅跳過測試
		- 計數顯示
		- README fallback
	completed_at: 2026-03-06
	deliverables:
		- services/kg/relationships.py (build_cosine_relationships with skip)
# TASK-062d 治理
governance:
	status: Completed
	owner: platform-kg@team
	priority: P2
	estimate: 1p
	risk: "調參缺乏指引"
	mitigation: "JSON 報告聚合測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 報告含 count/平均
		- 調參文件
		- 腳本說明
	completed_at: 2026-03-06
	deliverables:
		- scripts/kg_threshold_tune.py
```

```yaml
# TASK-065 治理
governance:
	status: Completed
	owner: platform-kg@team
	priority: P2
	estimate: 2p
	risk: "圖形可視化 bundle 體積膨脹"
	mitigation: "Lazy chunk + 體積 diff 測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- 動態載入驗證
		- 體積差異記錄
		- README 可視化章節
	completed_at: 2026-03-06
	notes: "KgPanel.tsx 已加入 window.ENABLE_KG_PANEL feature flag；Cytoscape lazy import 延後至 Sprint 6"
```

### 5.7 WebSocket 與即時
| ID       | Title               | Description          | Acceptance Criteria                                                                      | Dependencies      | Artifacts               | Req Mapping        |
|----------|---------------------|----------------------|------------------------------------------------------------------------------------------|-------------------|-------------------------|--------------------|
| TASK-070 | WS Gateway          | /ui/events multiplex | Handshake 成功；平均重連時間 <2s（基線工作負載）；心跳 15s 間隔允許連續 2 次 miss 才觸發降級 | TASK-036,TASK-044 | ws/gateway.py           | UI-FR-049~051      |
| TASK-071 | 事件封包與序列      | seq/heartbeat/gap    | 缺口觸發 resync                                                                          | TASK-070          | ws/envelope.py          | UI-FR-049~051, §23 |
| TASK-072 | useEventStream Hook | 管理連線/重試/派發   | 斷線恢復成功                                                                             | TASK-071          | insights-portal/hooks/* | UI-FR-049~051      |
| TASK-073 | 降級策略            | N 次失敗後回輪詢     | E2E 證明降級                                                                             | TASK-072          | insights-portal/hooks/* | UI-FR-049~051      |

```yaml
# TASK-070 治理
governance:
	status: Completed
	owner: platform-realtime@team
	priority: P1
	estimate: 2p
	risk: "無界重連導致資源風暴"
	mitigation: "退避 + 心跳降級測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- Handshake 測試
		- 心跳 miss 降級
		- 重連指標存在
	completed_at: 2026-03-06
	deliverables:
		- services/ws/gateway.py
		- services/tests/ws/test_gateway.py
```
```yaml
# TASK-071 治理
governance:
	status: Completed
	owner: platform-realtime@team
	priority: P1
	estimate: 2p
	risk: "序列缺口競態問題"
	mitigation: "決定性 gap 測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- 缺口偵測測試
		- 心跳間隔測試
		- Resync 回退測試
	engineer: E3
	target_sprint: 5
	completed_at: 2026-03-06
	deliverables:
		- services/ws/envelope.py
		- services/tests/ws/test_envelope.py
```

```yaml
# TASK-072 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 5
	completed_at: 2026-03-06
	deliverables:
		- insights-portal/src/hooks/useEventStream.ts
# TASK-073 治理
governance:
	status: Completed
	engineer: E3
	target_sprint: 5
	completed_at: 2026-03-06
	notes: "漸進式降級邏輯在 useEventStream.ts (MAX_CONSECUTIVE_FAILURES=5, DOWNGRADE_DURATION_MS=120000)"
```

### 5.8 Telemetry 與可觀測
| ID       | Title               | Description                 | Acceptance Criteria                | Dependencies | Artifacts                   | Req Mapping       |
|----------|---------------------|-----------------------------|------------------------------------|--------------|-----------------------------|-------------------|
| TASK-080 | Metrics 暴露        | Prometheus counters/hist    | /metrics 出現 ingestion/processing | TASK-010     | services/*/metrics.py       | NFR observability |
| TASK-081 | 前端 Telemetry      | logEvent + 批次             | 記錄 ui.kg.render / ui.ws.connect  | TASK-072     | insights-portal/telemetry/* | §25 taxonomy      |
| TASK-082 | Bundle/Perf 守門    | CI 檢查 chunk 體積          | 超標失敗 (KG chunk >300KB)         | TASK-065     | .github/workflows/ci.yml    | UI-NFR-006        |
| TASK-083 | Manifest 完整性原型 | 產生 manifest.json + sha256 | 清單校驗通過                       | TASK-034     | eval/manifest.py            | 建議提升          |
| TASK-084 | 事件 Schema 驗證    | WS envelope JSON Schema     | 無效事件被捨棄且計數               | TASK-071     | ws/schema.py                | 可靠性            |

```yaml
# TASK-080 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-observability@team
	priority: P1
	estimate: 1p
	risk: "缺少基礎指標隱藏回歸"
	mitigation: "指標存在性測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- /metrics 暴露計數器
		- 直方圖測試
		- README 指標列表
	deliverables:
		- services/common/metrics.py
		- services/tests/test_common_metrics.py

# TASK-082 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-observability@team
	priority: P2
	estimate: 1p
	risk: "Bundle 體積成長未察覺"
	mitigation: "CI 預算腳本 + diff 測試"
	adr_impact: []
	ci_gate: ["unit-tests","perf-baseline"]
	dod:
		- 預算失敗測試
		- Diff 報告 artifact
		- 文檔預算政策
	deliverables:
		- scripts/check_bundle_size.py
		- .github/workflows/bundle-size-guard.yml
		- services/tests/test_bundle_size.py

# TASK-083 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-observability@team
	priority: P2
	estimate: 1p
	risk: "Artifact 漂移未被偵測"
	mitigation: "Manifest schema + checksum 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- Manifest schema 測試
		- 缺少 artifact 失敗
		- README manifest 章節
	deliverables:
		- services/eval/manifest.py (generate_run_manifest)
		- services/tests/eval/test_manifest.py (5 new tests)
```
```yaml
# TASK-081 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-ui@team
	priority: P2
	estimate: 2p
	risk: "高頻事件過量"
	mitigation: "批次 & flush 間隔測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- 批次測試
		- 事件型別覆蓋清單
		- README telemetry 使用
	deliverables:
		- insights-portal/src/telemetry/logEvent.ts
		- insights-portal/src/telemetry/logEvent.test.ts
	engineer: E3
```
```yaml
# TASK-084 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	owner: platform-realtime@team
	priority: P1
	estimate: 1p
	risk: "無效事件污染狀態"
	mitigation: "Schema 驗證 + counter"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- 無效丟棄測試
		- 計數器遞增測試
		- 旗標開關文件
	engineer: E3
	target_sprint: 5
	deliverables:
		- services/ws/schema.py
		- services/tests/ws/test_schema.py
```

### 5.9 安全與隱私 (Phase 2 準備)
| ID       | Title           | Description                  | Acceptance Criteria | Dependencies | Artifacts           | Req Mapping   |
|----------|-----------------|------------------------------|---------------------|--------------|---------------------|---------------|
| TASK-090 | Auth 插槽       | 可插拔 token 驗證（dev no-op） | 設定 token 自動附加 | TASK-001     | common/auth.py      | 未來 NFR      |
| TASK-091 | PII 遮罩工具    | 日誌/摘要敏感遮罩            | 測試遮罩覆蓋 >95%   | TASK-003     | common/redact.py    | UI-FR-056/058 |
| TASK-092 | Rate Limit/背壓 | 基本 per-IP 限制             | 429 + Retry-After   | TASK-010     | common/ratelimit.py | 穩定性        |

```yaml
# TASK-090 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E1
	target_sprint: 5
	deliverables:
		- services/common/auth.py
		- services/tests/test_common_auth.py
# TASK-091 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E1
	target_sprint: 5
	deliverables:
		- services/common/redact.py
		- services/tests/test_common_redact.py
# TASK-092 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E1
	target_sprint: 5
	deliverables:
		- services/common/ratelimit.py
		- services/tests/test_common_ratelimit.py
```

### 5.10 測試與強化
| ID       | Title              | Description                       | Acceptance Criteria   | Dependencies      | Artifacts                | Req Mapping |
|----------|--------------------|-----------------------------------|-----------------------|-------------------|--------------------------|-------------|
| TASK-100 | Unit Coverage ≥70% | ingestion/processing/testset/eval | 報告 >=70% statements | Phases 1–4        | tests/*                  | 品質        |
| TASK-101 | E2E Pipeline Smoke | 最小 doc→report 腳本              | 報告生成且鏈完整      | TASK-044          | scripts/e2e_smoke.sh     | 追溯目標    |
| TASK-102 | 效能基線           | 測量 ingestion→eval latency       | baseline.json 產出    | TASK-101          | benchmarks/baseline.json | 效能        |
| TASK-103 | 負載測試           | k6/Locust 并發                    | p95 達標              | TASK-015,TASK-034 | load/*                   | 擴充性      |
| TASK-104 | 韌性混沌演練       | 模擬暫時失敗                      | 恢復無資料損壞        | TASK-016,TASK-033 | chaos/plan.md            | 可靠性      |

```yaml
# TASK-100 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	notes: "coverage 89% statements (閾值 ≥70%). 執行: pytest --cov=services"
# TASK-101 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- scripts/e2e_smoke.sh
# TASK-102 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- scripts/capture_perf_baseline.py
		- benchmarks/baseline.json
# TASK-103 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- load/locustfile.py
		- load/README.md
# TASK-104 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- chaos/plan.md
```

### 5.11 文件與營運就緒
| ID       | Title                      | Description                                                         | Acceptance Criteria            | Dependencies | Artifacts                              | Req Mapping |
|----------|----------------------------|---------------------------------------------------------------------|--------------------------------|--------------|----------------------------------------|-------------|
| TASK-110 | OpenAPI 草稿               | 產出並提交 openapi.json                                             | 每服務 openapi.json 存在       | TASK-010+    | services/*/openapi.json                | DevEx       |
| TASK-111 | Runbook                    | Ingestion/Processing/Eval 運維指引                                  | Markdown 含告警/儀表連結       | TASK-080     | docs/runbooks/*.md                     | Ops         |
| TASK-112 | 部署 Manifests             | Helm chart / k8s yaml                                               | helm dry-run 成功              | TASK-001     | deploy/helm/*                          | 部署        |
| TASK-113 | ADR 完成                   | ADR-001..004 文件化                                                 | 被 design 引用                 | 決策輸入     | docs/adr/*.md                          | 治理        |
| TASK-114 | UI 開發指南                | Portal README 加入 lifecycle                                        | README 更新通過 lint           | TASK-017     | insights-portal/README.md              | DevEx       |
| TASK-115 | ADR 擴充 005-006           | Telemetry 分類與事件 Schema 版本策略 ADR 文件                       | ADR-005/006 存在且雙語         | TASK-113     | docs/adr/ADR-005*, ADR-006*            | 治理        |
| TASK-116 | UI 中文 ADR 交叉引用       | `design.ui.zh.md` 新增 ADR 表                                       | 表含 ADR-001..006 且狀態一致   | TASK-115     | design.ui.zh.md                        | 治理        |
| TASK-117 | ADR 狀態提升 Accepted 準備 | 原型驗證後將 ADR-001..006 狀態變更為 Accepted 並註記驗證依據        | ADR 更新 & design 引用狀態同步 | 原型驗證輸出 | docs/adr/ADR-00*                       | 治理        |
| TASK-118 | 事件 Schema Registry 建立  | 建立 events/schemas/*.json 與 registry.json (name, version, sha256) | registry 驗證腳本通過          | TASK-084     | events/schemas/*, events/registry.json | 治理        |
| TASK-119 | Telemetry Taxonomy JSON    | 產出 telemetry_taxonomy.json + 驗證腳本                             | taxonomy 驗證通過 & 無重複 key | TASK-081     | telemetry/telemetry_taxonomy.json      | 治理        |

```yaml
# TASK-110 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- services/processing/openapi.json
		- services/testset/openapi.json
		- services/eval/openapi.json
		- services/reporting/openapi.json
		- services/adapter/openapi.json
		- services/kg/openapi.json
		- services/ws/openapi.json
		- scripts/generate_openapi_specs.py
# TASK-111 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- docs/runbooks/ingestion.md
		- docs/runbooks/processing.md
		- docs/runbooks/eval.md
# TASK-112 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- deploy/helm/Chart.yaml
		- deploy/helm/values.yaml
		- deploy/helm/templates/
# TASK-113 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- docs/adr/ADR-001-microservices-structure.md
		- docs/adr/ADR-002-knowledge-graph-visualization-tech.md
		- docs/adr/ADR-003-subgraph-sampling-strategy.md
		- docs/adr/ADR-004-manifest-integrity-and-artifact-traceability.md
# TASK-114 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E2
	target_sprint: 5
	deliverables:
		- insights-portal/README.md（Lifecycle Module 章節）
# TASK-115 治理
governance:
	status: Completed
	completed_at: 2026-03-06
	engineer: E2
	target_sprint: 5
	deliverables:
		- docs/adr/ADR-005-telemetry-taxonomy.md
		- docs/adr/ADR-005-telemetry-taxonomy.zh.md
		- docs/adr/ADR-006-event-schema-versioning.md
		- docs/adr/ADR-006-event-schema-versioning.zh.md
# TASK-116 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E2
	target_sprint: 5
	deliverables:
		- eval-pipeline/docs/design/design.ui.zh.md（附錄 A ADR 交叉引用表）
# TASK-117 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E2
	target_sprint: 5
	deliverables:
		- docs/adr/ADR-001.md ～ ADR-006.md（Status: Accepted 確認）
# TASK-118 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E2
	target_sprint: 5
	deliverables:
		- events/schema_registry.json
		- events/schemas/*.json
		- scripts/validate_event_schemas.py
		- services/tests/test_schema_validators.py
# TASK-119 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E2
	target_sprint: 5
	deliverables:
		- telemetry/telemetry_taxonomy.json
		- scripts/validate_telemetry_taxonomy.py
		- services/tests/test_schema_validators.py
```

### 5.12 容器化與部署強化 (Enhancements)
| ID       | Title                         | Description                                                                                                   | Acceptance Criteria                                                                       | Dependencies       | Artifacts                                  | 理由 / 對應           |
|----------|-------------------------------|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------------------|--------------------------------------------|-----------------------|
| TASK-120 | 多服務 Compose 基線           | 將現行單容器拆分為 ingestion/processing/testset/eval/reporting/ws/kg 多服務，新增 docker-compose.services.yml。 | docker-compose.services.yml 啟動全部服務；各服務健康檢查通過；日誌分離；共享 network 與 env。 | TASK-001, TASK-070 | docker-compose.services.yml                | 部署可擴充性, ADR-001 |
| TASK-121 | 開發熱重載與原始碼 Bind Mount | dev 覆寫檔啟用 src bind mount + uvicorn --reload。                                                             | docker-compose.dev.override.yml 下程式碼改動 <3 秒反映；DOCKER_README 增說明。              | TASK-120           | docker-compose.dev.override.yml, docs 更新 | 開發效率 DevEx        |

```yaml
# TASK-120 治理
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	engineer: E1
	artifacts:
		- docker-compose.services.yml
		- docker-compose.dev.override.yml
		- docs/DOCKER_README.md
		- docs/deployment_guide.md
	dod:
		- 服務分離並可透過 compose 啟動，健康檢查通過
		- dev override 啟用 bind mount 與 --reload，文件化流程
	completed_on: 2025-09-25
	verification:
		- docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml config
```
| TASK-122 | 映像版本與標籤策略            | 语义版號 + git SHA 標籤腳本與 Make 目標（build:tag）。                                                           | make build-tag 產生 :vX.Y.Z 與 :git-<sha>; docker images 顯示；strategy 文檔化。                                  | TASK-120           | scripts/tag_image.sh, Makefile 更新, docs/deployment_guide.md                       | 追溯 / 發布治理          |
| TASK-123 | CI 治理 + 映像建置流程        | GitHub Actions 執行 (schema + taxonomy 驗證) 後建置與推送標籤影像（main 分支）。                                 | build-governance.yml PR 通過；驗證失敗阻擋；成功推送影像紀錄。                                                     | TASK-118, TASK-119 | .github/workflows/build-governance.yml                                              | 自動化, 治理, 可靠性     |
| TASK-124 | 安全掃描整合                  | CI 整合 Trivy（或同等）；HIGH/CRITICAL 失敗；輸出 sarif artifact。                                                 | CI 顯示掃描步驟；注入測試漏洞導致失敗；sarif artifact 可下載。                                                     | TASK-123           | workflow 擴充, docs/security.md                                                     | 安全 NFR                 |
| TASK-125 | 基底映像強化與非 root         | Dockerfile：非 root 使用者、層精簡、pip --no-cache-dir、支援 pip mirror/離線守門 build args、cache volume 參數。        | docker history 顯示層減少；user != 0；移除套件管理快取；最終層數 <12；離線建置時日誌顯示 PyPI skip guard；尺寸差異記錄；提供 hardening checklist。 | TASK-120           | Dockerfile 更新, docs/deployment_guide.md                                           | 安全 + 效能              |
| TASK-126 | 擴充/外掛掛載模式             | 提供 extensions/ 目錄掛載 + 載入器，允許丟入 metrics / builder 外掛免重建。                                     | 放入 sample 外掛成功載入；README 說明；單元測試列舉外掛。                                                          | TASK-032, TASK-120 | extensions/sample_metric.py, services/common/plugin_loader.py                       | 可擴充性, ADR-001 模組化 |
| TASK-127 | Helm Chart 拆解               | 建立 Helm Chart，子模板對應各服務 + values 切換（kg, ws enable/disable）。                                        | helm template 成功；kg.enabled=false 排除 KG；README 說明 values。                                                 | TASK-120           | deploy/helm/Chart.yaml, deploy/helm/templates/*                                     | K8s 就緒, 部署彈性       |
| TASK-128 | 健康 / 就緒探針標準化         | /healthz /readyz endpoint + compose/helm 探針；heavy init (embedding) startup probe。                           | 各服務 200 回應；故意失敗測試非 ready；compose 與 Helm 定義探針。                                                  | TASK-120, TASK-127 | services/*/health.py, helm 模板更新, compose 更新                                   | 可靠性, 營運             |
| TASK-129 | K8s 水平擴展策略 (HPA)        | 為無狀態服務提供 CPU + 請求速率 HPA 範例。                                                                     | hpa.yaml 套用成功；dry-run 模擬 scaling；docs 含調優指引。                                                         | TASK-127, TASK-080 | deploy/helm/templates/hpa.yaml, docs/scaling.md                                     | 擴展性 NFR               |
| TASK-130 | SBOM 與映像簽章流水線         | CI 使用 syft 產出 SBOM，並（選）cosign 簽章與 provenance attestation。                                            | build-governance 產出 CycloneDX v1.5 SBOM 於 sbom/sbom-main.json + 簽章（若提供金鑰）；驗證步驟通過；artifact 保存。 | TASK-123, TASK-124 | .github/workflows/build-governance.yml, sbom/sbom-main.json, docs/security.md       | 供應鏈完整性             |
| TASK-131 | GPU 選配建置與執行設定        | ENABLE_GPU build arg + compose/Helm profile；曝露 gpu_enabled metric + 文件 fallback。                          | GPU profile 成功建置不破壞 CPU；gpu_enabled metric 存在；文件描述啟用步驟。                                        | TASK-125           | Dockerfile (ARG), docs/deployment_guide.md, helm values                             | 效能彈性                 |
| TASK-132 | 開發/CI 環境一致性驗證腳本    | 腳本比對 Python 版本、依賴鎖定、extensions 哈希；可做為 CI Gate。                                                 | 漂移退出碼非 0；附示範測試；design §21.12 引用。                                                                   | TASK-120           | scripts/validate_dev_parity.py, docs/deployment_guide.md                            | 可重現性, DevEx          |
| TASK-133 | Policy as Code (OPA)          | 使用 OPA Rego 規則驗證事件 schema 與 metrics 命名（前綴/風格/保留字）並納入 CI gate。                            | policy/*.rego 存在；命名違規測試失敗；CI 含 policy 驗證步驟。                                                      | TASK-118, TASK-032 | policy/*.rego, scripts/validate_policies.sh, .github/workflows/build-governance.yml | 治理, 一致性             |
| TASK-134 | Secrets Scan Gate (gitleaks)  | 整合 gitleaks 掃描新提交/PR 硬編碼 secrets；偵測即失敗（可允許清單）。                                            | CI 顯示 gitleaks 步驟；注入測試 secret 觸發失敗；允許清單文件化。                                                  | TASK-123           | .github/workflows/build-governance.yml, .gitleaks.toml, docs/security.md            | 安全, 供應鏈控制         |

```yaml
# TASK-122 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 5
	artifacts:
		- scripts/tag_image.sh
		- VERSION
	dod:
		- 標籤腳本可產生 vX.Y.Z 與 git-<sha>
		- deployment_guide.md 已描述策略
# TASK-123 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-124 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-125 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-126 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-26
# TASK-127 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- deploy/helm/Chart.yaml
		- deploy/helm/values.yaml
		- deploy/helm/templates/deployment.yaml
		- deploy/helm/templates/service.yaml
		- deploy/helm/templates/hpa.yaml
		- deploy/helm/templates/configmap.yaml
		- deploy/helm/templates/_helpers.tpl
# TASK-128 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- services/*/main.py (/healthz + /readyz 端點)
		- services/ws/gateway.py (/healthz + /readyz 端點)
		- deploy/helm/values.yaml (readinessProbe: /readyz, livenessProbe: /healthz)
# TASK-129 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- docs/scaling.md (HPA 調校指南)
		- deploy/helm/templates/hpa.yaml (已在 Sprint-6 完成)
# TASK-130 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- .github/workflows/build-governance.yml (syft SBOM + anchore/sbom-action)
		- sbom/ 目錄佔位
# TASK-131 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- Dockerfile (ARG ENABLE_GPU=false，條件式 CUDA torch 安裝)
		- deploy/helm/values.yaml (gpu: 區塊含 processing/kg GPU profiles)
# TASK-132 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- scripts/validate_dev_parity.py (Python 版本 + 套件存在性檢查)
# TASK-133 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- policy/naming.rego (事件鍵 + 指標名稱命名規則)
		- policy/schema_registry.rego (Registry 完整性)
# TASK-134 治理
governance:
	status: Completed
	completed_at: 2026-03-07
	engineer: E1
	target_sprint: 6
	deliverables:
		- .gitleaks.toml (允許清單設定)
		- .github/workflows/build-governance.yml (secrets-scan job)
# TASK-122 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-123 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-124 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-125 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-25
# TASK-126 治理
governance:
	status: Verified
	engineer: E1
	target_sprint: 6
	completed_on: 2025-09-26
# TASK-127 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-128 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-129 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-130 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-131 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-132 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-133 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
# TASK-134 治理
governance:
	status: Planned
	engineer: E1
	target_sprint: 6
```

```yaml
# TASK-120 治理
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	risk: "多服務拆分造成設定漂移"
	mitigation: "Compose 驗證腳本 + 健康矩陣測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	artifacts:
		- docker-compose.services.yml
		- .env.compose
		- docs/deployment_guide.md
	dod:
		- docker compose config 成功（預設環境模板）
		- 健康檢查端點維持可用
		- 部署指南補充環境覆寫說明
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 docker compose -f docker-compose.services.yml config

# TASK-123 治理
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	risk: "建置流程跳過治理驗證"
	mitigation: "工作流程步驟排序測試 + 失敗注入"
	adr_impact: ["ADR-004","ADR-005"]
	ci_gate: ["build-governance:schemas"]
	artifacts:
		- .github/workflows/build-governance.yml
		- scripts/validate_task_status.py
		- scripts/validate_compose.py
	dod:
		- 工作流程先執行治理驗證再建置映像
		- 任務狀態飄移可被 validate_task_status.py 阻擋
		- main 分支建置時推送 GHCR 標籤
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 python3 scripts/validate_task_status.py

# TASK-124 治理
governance:
	status: Verified
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "Critical 漏洞未被攔截"
	mitigation: "注入 CVE 測試觸發失敗"
	adr_impact: ["ADR-004"]
	ci_gate: ["security-scan"]
	artifacts:
		- .github/workflows/build-governance.yml
		- docs/security.md
	dod:
		- Trivy 檔案系統與映像掃描產出 SARIF
		- HIGH/CRITICAL 漏洞觸發 exit-code 1
		- 安全指南記錄修復與豁免流程
	completed_on: 2025-09-25
	verification:
		- Workflow 審閱：Trivy 步驟已啟用並上傳 SARIF

# TASK-125 治理
governance:
	status: Verified
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "基底映像 CVE 未及時修補或仍以 root 執行"
	mitigation: "多階段 Dockerfile + 安全掃描管線定期檢視"
	adr_impact: ["ADR-004"]
	ci_gate: ["unit-tests"]
	artifacts:
		- Dockerfile
		- docker-compose.services.yml
		- .env.compose
		- docs/deployment_guide.md
		- docs/hardening_checklist.md
	dod:
		- Dockerfile 改為 builder/runtime 多階段，最終階段不含建置工具鏈
		- 維持非 root 用戶並確保 `/app`、`${MODELS_CACHE_PATH}`、`${EXTENSIONS_DIR}` 權限歸屬
		- 依賴安裝集中於單一 RUN，啟用 `pip --no-cache-dir`、`PIP_DISABLE_PIP_VERSION_CHECK`、停用 bytecode
		- 新增 pip mirror / 離線守門 build arg，於 Hardening 清單中記錄避免離線重試暴增
		- 基底映像更新為 `python:3.11-slim-bookworm` 並於各階段執行 `apt-get upgrade`，MODELS_CACHE 參數仍可與 compose volume 對應
		- Hardening 清單與部署指南同步描述驗證步驟與多階段理由
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 docker build -t rag-eval:test .
		- 2025-09-25 docker history rag-eval:test | head -n 12
		- 2025-09-25 grep "python:3.11-slim-bookworm" Dockerfile
		- 2025-09-25 python3 -m compileall services

# TASK-126 治理
governance:
	status: Verified
	owner: platform-extensions@team
	priority: P2
	estimate: 2p
	risk: "外掛載入流程缺乏覆蓋，可能在回歸時失效"
	mitigation: "針對 register() 與屬性回退的單元測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests"]
	artifacts:
		- services/common/plugin_loader.py
		- extensions/sample_metric.py
		- test_plugin_loader.py
		- docs/DOCKER_README.md
		- docs/deployment_guide.md
	dod:
		- 載入器可發現 register() 與 PLUGIN_DEFINITION 外掛
		- 範例外掛可直接掛載於 extensions/ 免重建
		- 文件說明 EXTENSIONS_DIR 覆寫與使用方式
		- 自動化測試列舉外掛並驗證輸出
	completed_on: 2025-09-26
	verification:
		- 2025-09-26 pytest test_plugin_loader.py

# TASK-122 治理
governance:
	status: Verified
	owner: platform-deploy@team
	priority: P1
	estimate: 1p
	risk: "標籤策略不一致"
	mitigation: "標籤腳本 snapshot 測試"
	adr_impact: ["ADR-004"]
	ci_gate: ["unit-tests"]
	artifacts:
		- scripts/tag_image.sh
		- Makefile
		- VERSION
		- docs/deployment_guide.md
	dod:
		- tag_image.sh 同時輸出 v<version> 與 git-<sha>（支援 DRY_RUN）
		- VERSION 仍為語義版號權威
		- 部署指南補充 build-tag 流程
	completed_on: 2025-09-25
	verification:
		- 2025-09-25 DRY_RUN=1 make tag
	engineer: E1
	target_sprint: 5
# TASK-127 治理
governance:
	status: Planned
	owner: platform-deploy@team
	priority: P1
	estimate: 2p
	risk: "單一 chart 複雜度過高"
	mitigation: "模板 linter + values 測試"
	adr_impact: ["ADR-004"]
	ci_gate: ["unit-tests"]
	dod:
		- helm template 測試
		- kg/ws disabled diff 測試
		- README values 表格
	engineer: E1
	target_sprint: 5
# TASK-128 治理
governance:
	status: Planned
	owner: platform-deploy@team
	priority: P1
	estimate: 1p
	risk: "缺 readiness 隱藏故障"
	mitigation: "探針失敗夾具"
	adr_impact: ["ADR-004"]
	ci_gate: ["unit-tests"]
	dod:
		- /healthz 測試
		- /readyz 測試
		- Startup probe 文件
	engineer: E1
	target_sprint: 5
# TASK-129 治理
governance:
	status: Planned
	owner: platform-deploy@team
	priority: P2
	estimate: 1p
	risk: "HPA 過度或不足 scaling"
	mitigation: "HPA dry-run 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- HPA 範例套用測試
		- Scaling 事件文件
		- 調優指南
	engineer: E1
	target_sprint: 6
# TASK-131 治理
governance:
	status: Planned
	owner: platform-deploy@team
	priority: P3
	estimate: 1p
	risk: "GPU 路徑與 CPU 分歧"
	mitigation: "Parity build 測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- gpu_enabled metric 測試
		- CPU fallback 測試
		- README GPU 章節
	engineer: E1
	target_sprint: 6
# TASK-132 治理
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "環境漂移降低重現性"
	mitigation: "Parity 腳本 diff 測試"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- 漂移退出碼測試
		- JSON parity 報告
		- README parity 章節
 	engineer: E1
 	target_sprint: 6
```yaml
# TASK-133 治理
governance:
	status: Planned
	owner: platform-governance@team
	priority: P2
	estimate: 2p
	risk: "政策缺口導致指標不一致"
	mitigation: "命名違規負向測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["policy-validate"]
	dod:
		- Rego 測試通過
		- 命名違規樣本
		- README policy 章節
	engineer: E1 (E2 metrics 諮詢)
	target_sprint: 6
# TASK-134 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "Secrets 泄漏入庫"
	mitigation: "注入 secret 測試"
	adr_impact: ["ADR-004"]
	ci_gate: ["security-scan"]
	dod:
		- gitleaks 設定提交
		- 失敗測試樣本
		- Allowlist 文件
	engineer: E1
	target_sprint: 5
```
```

```yaml
# TASK-130 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 4p
	risk: "簽章金鑰管理複雜或遺失導致部署阻塞"
	mitigation: "允許無金鑰時降級為 unsigned (標註)；KMS 托管金鑰 + 最小權限"
	adr_impact: ["ADR-004"]
	ci_gate: ["security-scan","sbom-generate","image-sign"]
	sbom_format: CycloneDX-1.5
	artifacts:
		- sbom/sbom-main.json
		- sbom/sbom-diff.json
		- attest/provenance.intoto.jsonl
	slo:
		pipeline_additional_time_seconds_p95: 90
		critical_vuln_allowed: 0
	metrics:
		- supplychain_vuln_count{severity="CRITICAL"}
		- supplychain_unsigned_image_total
	logs:
		- code=SBOM_GENERATED level=INFO
		- code=IMAGE_SIGNED    level=INFO
		- code=SIGNING_SKIPPED level=WARN
	dod:
		- syft 產出 CycloneDX JSON 並存入 sbom/
		- Trivy 掃描 SARIF 上傳並零 CRITICAL/HIGH (允許 X 個中等)
		- 提供 cosign 驗證示例指令於 security.md
		- 簽章缺失時 workflow 標註 annotation
		- README 部署章節新增 SBOM / 簽章說明
```

#### TASK-126 子任務分解
| 子ID      | 標題                | 說明                                     | 驗收條件                             | 依賴      | 產出                              | 備註     |
|-----------|---------------------|------------------------------------------|--------------------------------------|-----------|-----------------------------------|----------|
| TASK-126a | 目錄監控 Reload(選) | dev 模式檔案變更自動載入外掛             | 新增/刪除檔案 dev 立即反映；prod 關閉 | TASK-126  | services/common/plugin_loader.py  | 開發效率 |
| TASK-126b | Sandbox & Allowlist | 限制外掛匯入 allowlist                   | 禁用匯入拋錯並記錄安全事件           | TASK-126a | services/common/plugin_sandbox.py | 安全     |
| TASK-126c | 版本協商            | 讀取外掛 manifest (version/capabilities) | 不相容版本跳過並警告                 | TASK-126b | extensions/manifest.schema.json   | 相容性   |
| TASK-126d | 失敗 Telemetry 事件 | 發佈 plugin.failed / plugin.loaded 事件  | 日誌與 metrics 計數更新              | TASK-126c | services/common/plugin_events.py  | 可觀測   |

```yaml
# TASK-126a 治理
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "熱重載造成記憶體洩漏"
	mitigation: "壓力重載測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 新增/刪除檔即載入
		- 生產停用驗證
		- README 重載章節
# TASK-126b 治理
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P1
	estimate: 1p
	risk: "未信任 import 執行任意碼"
	mitigation: "allowlist + sandbox 測試"
	adr_impact: ["ADR-001"]
	ci_gate: ["unit-tests","security-scan"]
	dod:
		- 阻擋 import 測試
		- 安全事件日誌
		- README sandbox
# TASK-126c 治理
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "不相容版本仍載入"
	mitigation: "版本協商測試"
	adr_impact: []
	ci_gate: ["unit-tests"]
	dod:
		- 跳過不相容測試
		- 警告日誌
		- Manifest schema 文件
# TASK-126d 治理
governance:
	status: Planned
	owner: platform-extensions@team
	priority: P2
	estimate: 1p
	risk: "外掛失敗不可見"
	mitigation: "事件 + 指標測試"
	adr_impact: ["ADR-005"]
	ci_gate: ["unit-tests"]
	dod:
		- 失敗事件測試
		- 指標遞增
		- README telemetry
```

#### TASK-130 子任務分解
| 子ID      | 標題               | 說明                                    | 驗收條件                               | 依賴      | 產出                           | 備註      |
|-----------|--------------------|-----------------------------------------|----------------------------------------|-----------|--------------------------------|-----------|
| TASK-130a | SBOM 生成          | syft 產出 CycloneDX JSON                | 生成 sbom-main.json 並 schema 驗證     | TASK-130  | sbom/sbom-main.json            | 基線      |
| TASK-130b | 漏洞差異           | 比對當前 vs 前一次掃描輸出 diff         | diff JSON 列出新增 HIGH/CRITICAL       | TASK-130a | sbom/sbom-diff.json            | 治理 gate |
| TASK-130c | 條件簽章           | 有金鑰 cosign 簽；否則標註 unsigned      | 簽章影像 cosign verify 通過；無簽章警告 | TASK-130b | attest/*.intoto.jsonl          | 供應鏈    |
| TASK-130d | Attestation & 保留 | 產生 provenance 並清理舊 SBOM (≤N 保留) | provenance 存在；舊檔清理生效           | TASK-130c | attest/provenance.intoto.jsonl | 健康維護  |

```yaml
# TASK-130a 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "SBOM schema 漂移"
	mitigation: "Schema 驗證測試"
	adr_impact: ["ADR-004"]
	ci_gate: ["sbom-generate"]
	dod:
		- SBOM 檔存在
		- Schema 驗證通過
		- README SBOM 章節
# TASK-130b 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P1
	estimate: 1p
	risk: "漏洞 diff 漏掉新 critical"
	mitigation: "Diff 單元測試"
	adr_impact: []
	ci_gate: ["security-scan"]
	dod:
		- 新漏洞檢測測試
		- 退出碼邏輯
		- Docs diff 使用
# TASK-130c 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "未簽章影像未被察覺"
	mitigation: "verify 指令測試"
	adr_impact: []
	ci_gate: ["image-sign"]
	dod:
		- cosign verify 測試
		- 未簽章警告
		- 文件簽章說明
# TASK-130d 治理
governance:
	status: Planned
	owner: platform-secops@team
	priority: P2
	estimate: 1p
	risk: "保留清理失敗"
	mitigation: "保留視窗測試"
	adr_impact: []
	ci_gate: ["sbom-generate"]
	dod:
		- Prune 測試
		- Attestation 存在
		- README 保留說明
```

#### TASK-132 子任務分解
| 子ID      | 標題               | 說明                             | 驗收條件                        | 依賴      | 產出                           | 備註     |
|-----------|--------------------|----------------------------------|---------------------------------|-----------|--------------------------------|----------|
| TASK-132a | Python 版本比對    | 比對本地 vs 容器 major.minor     | 不一致非 0 退出 + JSON 報告欄位 | TASK-132  | scripts/validate_dev_parity.py | 漂移偵測 |
| TASK-132b | 依賴鎖雜湊比對     | 計算鎖檔 hash 與容器快照比較     | 變更產生 drift 標誌 (可白名單)  | TASK-132a | scripts/validate_dev_parity.py | 完整性   |
| TASK-132c | 擴充指紋           | Hash extensions/ 每檔並比較      | 新增/刪除/變更列於 diff         | TASK-132b | scripts/validate_dev_parity.py | 擴充管理 |
| TASK-132d | 漂移白名單與格式化 | 支援允許清單；輸出 MD + JSON 報告 | 報告含摘要表；白名單項標註       | TASK-132c | scripts/validate_dev_parity.py | 報告     |

```yaml
# TASK-132a 治理
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "Python 版本漂移未察覺"
	mitigation: "版本不一致測試"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- 版本差異測試
		- JSON 報告存在
		- README parity 章節
# TASK-132b 治理
governance:
	status: Planned
	owner: platform-parity@team
	priority: P1
	estimate: 1p
	risk: "鎖檔漂移未標記"
	mitigation: "雜湊比較測試"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- 漂移標誌測試
		- 白名單文件
		- diff 報告段落
# TASK-132c 治理
governance:
	status: Planned
	owner: platform-parity@team
	priority: P2
	estimate: 1p
	risk: "擴充變更未追蹤"
	mitigation: "指紋 diff 測試"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- 指紋報告
		- Diff 測試
		- README 指紋說明
# TASK-132d 治理
governance:
	status: Planned
	owner: platform-parity@team
	priority: P2
	estimate: 1p
	risk: "白名單隱藏真正漂移"
	mitigation: "白名單覆寫測試"
	adr_impact: []
	ci_gate: ["parity-validate"]
	dod:
		- 白名單註記
		- MD+JSON 輸出
		- README 白名單政策
```

## 6. 需求覆蓋摘要
- Ingestion/Processing FR：TASK-010..016  
- Testset FR-013~016：TASK-020..024  
- Evaluation FR-017~022：TASK-030..035  
- Reporting FR-037~040：TASK-041..044  
- KM Export FR-041/042：TASK-045..046  
- KG UI-FR-016~018：TASK-060..065  
- 即時 & 效能 UI-FR-049~055：TASK-070..073, TASK-082, TASK-084  

## 7. 任務欄位與治理規範 (新增)
為提升任務可追蹤性與治理一致性，建議於關鍵或高風險任務後附加 `governance` YAML 區塊；欄位定義如下：

| 欄位                    | 說明                | 指南                                              |
|-------------------------|---------------------|---------------------------------------------------|
| status                  | 任務生命週期狀態    | Planned / In-Progress / Blocked / Done / Verified |
| owner                   | 負責人或群組        | 群組郵件或 Github Team 名稱                       |
| priority                | 優先級              | P0 (核心阻塞) ~ P3 (低)                           |
| estimate                | 規模估點            | Story Points 或 人日 (保持一致)                   |
| risk                    | 主要風險            | 精煉一句含觸發條件                                |
| mitigation              | 對策                | 減緩或降級策略                                    |
| adr_impact              | 受影響 ADR          | 若需更新列出編號                                  |
| ci_gate                 | 對應 CI 工作名稱    | 影響 merge 的 gate                                |
| slo                     | 量化 SLO / KPI      | latency / error rate 等                           |
| metrics                 | 預期新增指標        | 明確指標名稱 (snake_case)                         |
| logs                    | 需新增關鍵日誌      | 使用 code= 前綴以利搜尋                           |
| artifacts               | 額外產出            | 非主要表格列出者                                  |
| plugin_contract_version | (外掛相關) API 版本 | 變更需語義版本策略                                |
| failure_isolation       | (外掛) 隔離策略     | e.g. try-except per plugin                        |
| sbom_format             | (供應鏈) SBOM 格式  | CycloneDX / SPDX                                  |
| dod                     | Definition of Done  | 一致的核對清單                                    |

通用 DoD 建議最少包含：
1. 測試：單元 + 邊界 + 失敗路徑；必要時整合測試
2. 可觀測：新增/更新 metrics、關鍵日誌、必要 trace 標籤
3. 文件：README / design / ADR (如適用) 同步
4. 安全 / 合規：不引入高風險 CVE；密鑰不寫入程式碼
5. 效能：未超過 baseline 120% (若適用)
6. 失敗降級：主要邏輯失敗不致整體崩潰 (外掛 / 非核心功能)

驗證自動化建議：後續可透過 `scripts/validate_tasks.py` 檢查 YAML 區塊欄位完整性與指標命名規範。

### 7.1 快速參考（必填 / 條件欄位）
| 欄位                    | 是否必填 | 條件說明                               |
|-------------------------|----------|----------------------------------------|
| status                  | 是       | 全部任務                               |
| owner                   | 是       | 負責團隊或個人                         |
| priority                | 是       | P0~P3；核心關鍵為 P0                    |
| estimate                | 是       | Story Points / 人日擇一統一            |
| risk                    | 是       | P0/P1 至少 1 條                        |
| mitigation              | 是       | 與 risk 對應                           |
| adr_impact              | 是       | 無則 []                                |
| ci_gate                 | 條件     | 若任務引入檢核 / build 流程需列出      |
| slo                     | 條件     | 延遲/可靠性敏感任務需列                |
| metrics                 | 條件     | 服務邏輯 / 執行路徑任務需列，純文件可省 |
| logs                    | 條件     | 執行路徑任務至少一個 code= 事件        |
| artifacts               | 選填     | 額外輸出                               |
| plugin_contract_version | 條件     | 外掛 / 擴充相關                        |
| failure_isolation       | 條件     | 外掛執行需列隔離策略                   |
| sbom_format             | 條件     | 供應鏈 / 安全任務                      |
| dod                     | 是       | 最低 DoD 核對清單                      |

### 7.2 YAML Skeleton 範本
```yaml
governance:
	status: Planned
	owner: <team-or-user>
	priority: P1
	estimate: 3p
	risk: "<主要風險>"
	mitigation: "<降級 / 迴避策略>"
	adr_impact: []
	ci_gate: ["unit-tests"]
	slo:
		<metric_name>: <target>
	metrics:
		- <operation_duration_seconds>
	logs:
		- code=<EVENT_CODE> level=INFO
	artifacts: []
	plugin_contract_version: 1
	failure_isolation: "try-except per plugin"
	sbom_format: CycloneDX-1.5
	dod:
		- 單元 + 失敗路徑測試
		- 指標已在 /metrics 中
		- 關鍵日誌出現
		- 文件 (README/design/ADR) 更新
		- 無新增 HIGH/CRITICAL CVE
		- 已驗證降級不影響主流程
```

### 7.3 CI Gate 命名建議
| 目的                 | 建議 Job 名稱            |
|----------------------|--------------------------|
| 單元 + Lint          | unit-tests               |
| 覆蓋率門檻           | coverage-check           |
| Schema/Taxonomy 驗證 | build-governance:schemas |
| 前端 Bundle 體積     | bundle-size-guard        |
| 安全掃描             | security-scan            |
| SBOM 生成            | sbom-generate            |
| 映像簽章             | image-sign               |
| 效能冒煙             | perf-baseline            |
| Dev/CI 環境一致      | parity-validate          |

### 7.4 狀態工作流
Planned → In-Progress → (Blocked ↔ In-Progress) → Done → Verified（驗證或審核證據後）。

Blocked 任務需於 PR 或 Issue 補充：阻塞原因 + 下一步動作。
- 隱私 UI-FR-056/058：TASK-091  
- 旗標 & Lazy (UI-NFR-006)：TASK-065, TASK-082  
- 追溯 SMART#4：010..016, 020..024, 030..035, 041..044, 045 + TASK-083  

## 7. 風險與緩解（擷取）
| 風險           | 影響        | 緩解                                 |
|----------------|-------------|--------------------------------------|
| KG 過大        | UI 卡頓     | TASK-065 節點上限 + TASK-066 子圖    |
| 事件遺失       | UI 資料陳舊 | TASK-071 序列 + TASK-072 重同步      |
| Artifact 重複  | 儲存成本    | TASK-015 冪等 + TASK-020 config hash |
| 指標回退未察覺 | 品質漂移    | TASK-082 CI + TASK-102 基線          |
| 子圖濫用       | 後端壓力    | TASK-066 速率限制 + 決定性採樣       |

## 8. 最佳化 / 強化建議
- Manifest 完整性：TASK-083 穩定後升級為簽章 manifest v2。  
- Schema Registry：擴展 TASK-084，對事件 schema 哈希釘選 + CI 漂移檢查。  
- Metrics 快取：利用 evaluation 決定性哈希重用 kpis.json。  
- 漸進式 WebSocket：在錯誤率 <0.5% 前保留輪詢回退（TASK-073）。  
- Persona/Scenario 深入分析：M3 後新增跨 run 覆蓋比較視圖。  

## 9. RACI（摘要）
| 項目         | R        | A        | C            | I          |
|--------------|----------|----------|--------------|------------|
| 服務實作     | 平台工程 | 平台主管 | QA, Security | 利害關係人 |
| UI Lifecycle | 前端工程 | 前端主管 | 平台工程     | 利害關係人 |
| KG 功能      | 數據工程 | 平台主管 | 前端工程     | 利害關係人 |
| Telemetry/WS | 平台工程 | 平台主管 | SRE          | 利害關係人 |
| 文件與 ADR   | 技術寫作 | 平台主管 | 各工程主管   | 全組       |

## 10. 驗收與變更控制
重大範圍變動需版本遞增並（若涉架構）建立 ADR；任務順序微調不需版本升級，但必須維持需求覆蓋完整性。

---
文件結束。
## 12. 子任務 → 主任務追蹤矩陣

彙總所有高影響主任務之已定義子任務 (a–d)，用於治理、排程與風險鏈審視。

### 12.1 英文矩陣 (English Reference)
| Subtask   | Parent   | Domain       | Focus                            | Key Artifact                                | Depends On  | Downstream / Consumer        | Key Acceptance Signal       |
|-----------|----------|--------------|----------------------------------|---------------------------------------------|-------------|------------------------------|-----------------------------|
| TASK-015a | TASK-015 | Processing   | Tokenization & boundaries        | processing/stages/tokenizer.py              | Parent init | 015b/015c                    | Stable spans                |
| TASK-015b | TASK-015 | Processing   | Chunk size/overlap rules         | processing/stages/chunk_rules.py            | 015a        | 015c/015d                    | Size & determinism          |
| TASK-015c | TASK-015 | Processing   | Embedding batch exec + retries   | processing/stages/embed_executor.py         | 015b        | 015d, evaluation             | Retry & breaker metrics     |
| TASK-015d | TASK-015 | Processing   | Persistence & integrity manifest | processing/stages/chunk_persist.py          | 015c        | Testset/Eval stages          | Hash/count parity           |
| TASK-032a | TASK-032 | Eval         | Plugin interface contract        | eval/metrics/interface.py                   | Parent init | 032b/032c, 126c              | Clear missing-method errors |
| TASK-032b | TASK-032 | Eval         | Baseline metrics impl            | eval/metrics/baseline/*.py                  | 032a        | Evaluation run (030–035)     | Deterministic scores        |
| TASK-032c | TASK-032 | Eval         | Discovery + failure isolation    | eval/metrics/loader.py                      | 032b        | Future plugins, 126*         | Faulty plugin isolation     |
| TASK-062a | TASK-062 | KG           | Node property enrichment         | kg/extract.py                               | Parent init | 062b/062c/summary            | Properties completeness     |
| TASK-062b | TASK-062 | KG           | Jaccard & Overlap relations      | kg/relationships.py                         | 062a        | 062c/062d                    | >0 relationships sample     |
| TASK-062c | TASK-062 | KG           | Cosine similarity + fallback     | kg/relationships.py                         | 062b        | 062d tuning                  | Graceful no-embed skip      |
| TASK-062d | TASK-062 | KG           | Threshold tuning harness         | scripts/kg_threshold_tune.py                | 062c        | KG ops / threshold decisions | JSON metrics output         |
| TASK-126a | TASK-126 | Extensions   | Dev reload (optional)            | services/common/plugin_loader.py            | Parent init | 126b/126c                    | Live reload works           |
| TASK-126b | TASK-126 | Extensions   | Sandbox & allowlist              | services/common/plugin_sandbox.py           | 126a        | 126c/126d                    | Blocked disallowed import   |
| TASK-126c | TASK-126 | Extensions   | Version negotiation              | extensions/manifest.schema.json             | 126b        | 032a interface alignment     | Incompatible skipped        |
| TASK-126d | TASK-126 | Extensions   | Failure telemetry events         | services/common/plugin_events.py            | 126c        | Observability / SRE          | Events & counters present   |
| TASK-130a | TASK-130 | Supply Chain | SBOM generation                  | sbom/sbom-main.json                         | Parent init | 130b/130d                    | Valid CycloneDX             |
| TASK-130b | TASK-130 | Supply Chain | Vulnerability diff               | sbom/sbom-diff.json                         | 130a        | 130c signing decision        | New HIGH/CRITICAL flagged   |
| TASK-130c | TASK-130 | Supply Chain | Conditional signing              | attest/provenance.intoto.jsonl & signatures | 130b        | Deployment pipeline          | Verified or warning logged  |
| TASK-130d | TASK-130 | Supply Chain | Attestation & retention          | attest/provenance.intoto.jsonl              | 130c        | Audit & compliance           | Old artifacts pruned        |
| TASK-132a | TASK-132 | Parity       | Python version compare           | scripts/validate_dev_parity.py              | Parent init | 132b/132c/132d               | Exit code on mismatch       |
| TASK-132b | TASK-132 | Parity       | Lock file hash compare           | scripts/validate_dev_parity.py              | 132a        | 132d                         | Drift flag set              |
| TASK-132c | TASK-132 | Parity       | Extension fingerprint            | scripts/validate_dev_parity.py              | 132b        | 132d                         | Diff enumerated             |
| TASK-132d | TASK-132 | Parity       | Whitelist & formatted report     | scripts/validate_dev_parity.py              | 132c        | CI gate / reviewers          | Whitelisted drifts tagged   |

### 12.2 中文矩陣 (Chinese)
| 子任務    | 主任務   | 功能領域     | 核心焦點            | 主要產出/檔案                       | 直接依賴   | 下游/被誰使用     | 驗收核心指標    |
|-----------|----------|--------------|---------------------|-------------------------------------|------------|-------------------|-----------------|
| TASK-015a | TASK-015 | Processing   | Tokenizer + 邊界    | processing/stages/tokenizer.py      | 主任務啟動 | 015b/015c         | 邊界穩定        |
| TASK-015b | TASK-015 | Processing   | Chunk 規則/重疊     | processing/stages/chunk_rules.py    | 015a       | 015c/015d         | 無超限 & 決定性 |
| TASK-015c | TASK-015 | Processing   | 嵌入批次 + 重試     | processing/stages/embed_executor.py | 015b       | 015d, 評估        | 失敗/斷路指標   |
| TASK-015d | TASK-015 | Processing   | 持久化 + Manifest   | processing/stages/chunk_persist.py  | 015c       | Testset/Eval      | 雜湊/計數一致   |
| TASK-032a | TASK-032 | Eval         | 外掛介面契約        | eval/metrics/interface.py           | 主任務啟動 | 032b/032c, 126c   | 缺方法即錯      |
| TASK-032b | TASK-032 | Eval         | 基礎指標實作        | eval/metrics/baseline/*.py          | 032a       | 評估流程          | 決定性分數      |
| TASK-032c | TASK-032 | Eval         | 探索 + 隔離         | eval/metrics/loader.py              | 032b       | 未來外掛, 126*    | 壞外掛不連鎖    |
| TASK-062a | TASK-062 | KG           | 節點屬性增豐        | kg/extract.py                       | 主任務啟動 | 062b/062c/summary | 屬性完整率      |
| TASK-062b | TASK-062 | KG           | Jaccard & Overlap   | kg/relationships.py                 | 062a       | 062c/062d         | >0 關係樣本     |
| TASK-062c | TASK-062 | KG           | Cosine + fallback   | kg/relationships.py                 | 062b       | 062d              | 無向量仍通過    |
| TASK-062d | TASK-062 | KG           | 閾值調參工具        | scripts/kg_threshold_tune.py        | 062c       | KG 連線/閾值決策  | JSON 報告       |
| TASK-126a | TASK-126 | Extensions   | Dev 目錄熱重載      | services/common/plugin_loader.py    | 主任務啟動 | 126b/126c         | 變更即載入      |
| TASK-126b | TASK-126 | Extensions   | Sandbox Allowlist   | services/common/plugin_sandbox.py   | 126a       | 126c/126d         | 禁匯入阻擋      |
| TASK-126c | TASK-126 | Extensions   | 版本協商            | extensions/manifest.schema.json     | 126b       | 032a 介面對齊     | 不相容跳過      |
| TASK-126d | TASK-126 | Extensions   | 失敗 telemetry 事件 | services/common/plugin_events.py    | 126c       | 可觀測/SRE        | 事件+計數       |
| TASK-130a | TASK-130 | Supply Chain | SBOM 生成           | sbom/sbom-main.json                 | 主任務啟動 | 130b/130d         | CycloneDX 驗證  |
| TASK-130b | TASK-130 | Supply Chain | 漏洞差異            | sbom/sbom-diff.json                 | 130a       | 130c 簽章         | 新高風險標示    |
| TASK-130c | TASK-130 | Supply Chain | 條件簽章            | attest/provenance.intoto.jsonl 等   | 130b       | 部署流程          | 簽章驗證/警告   |
| TASK-130d | TASK-130 | Supply Chain | Attestation + 保留  | attest/provenance.intoto.jsonl      | 130c       | 稽核/合規         | 舊檔清理        |
| TASK-132a | TASK-132 | Parity       | Python 版本比對     | scripts/validate_dev_parity.py      | 主任務啟動 | 132b/132c/132d    | 不一致退出碼    |
| TASK-132b | TASK-132 | Parity       | 依賴鎖雜湊          | scripts/validate_dev_parity.py      | 132a       | 132d              | 漂移標誌        |
| TASK-132c | TASK-132 | Parity       | 擴充指紋            | scripts/validate_dev_parity.py      | 132b       | 132d              | 差異列出        |
| TASK-132d | TASK-132 | Parity       | 白名單 & 報告       | scripts/validate_dev_parity.py      | 132c       | CI Gate / 審閱    | 白名單標註      |

### 12.3 綜合觀察
最高依賴鏈：TASK-015（支援 testset/eval）。供應鏈風險路徑：130a→130b→130c→130d。外掛安全路徑：126a→126b→126c→126d（關聯 032a 契約）。KG 調參迴圈：062a→062b→062c→062d。

### 12.4 自動化建議
1. 腳本解析 `tasks*.md` 重新生成本矩陣 (Markdown + JSON) 以偵測漂移。
2. CI Gate：驗證每個子任務之 parent 存在且 governance skeleton 存在。
3. 覆蓋指標：高影響主任務拆解覆蓋率（目前選定集合 100%）。

