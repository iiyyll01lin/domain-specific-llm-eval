# UI 技術設計 – 評估工作流程主控台

版本：0.1  
狀態：徵求審閱  
日期：2025-09-10  
負責：平台工程組  

---
## 1. 目的與範圍
本文件將 UI EARS 需求（`../requirements/requirements.ui.zh.md` 與英文對應）轉化為具體技術架構與實作策略：組件邊界、資料與控制流程、狀態管理、功能旗標整合、錯誤處理、效能策略、測試方案與未來擴充。

非目標（Phase 1）：完整 RBAC、離線優先完整支持、衍生資源 CRUD UI、視覺細緻化指南、最終 A11y 報告。

## 2. 架構總覽
### 2.1 高階概念
本主控台以既有 Portal（React + Vite）內嵌模組方式提供生命週期導航（Documents → Processing → KG → Testsets → Evaluations → Insights → Reports → Admin），並以功能旗標控制 KG 可視化與多執行比較。所有後端互動透過服務 API 或標準化 artifacts 讀取（物件儲存），不新增額外專屬中介層。

### 2.2 執行期組件圖
```
+---------------- Portal Shell (既有) ----------------+
| Theme/I18n | FeatureFlagsProvider | Router           |
|            |         |             |                 |
|            v         v             v                 |
|        UI Lifecycle Module（本設計）                |
|  +-----------+ +-----------+ +-----------+           |
|  | Documents | |Processing | |   KG      | ...       |
|  +-----------+ +-----------+ +-----------+           |
|       |            |             |                   |
|       | REST(fetch)|             | REST + Artifacts  |
|       v            v             v                   |
| ingestion-svc  processing-svc  kg-builder-svc         |
|       |            |             |                   |
|  object storage vector store  KG JSON summaries       |
+------------------------------------------------------+
```

### 2.3 核心原則
- 輕量協調：UI 不自行流程編排，只發送工作與觀察狀態。
- 宣告式狀態：共用 store（沿用 Portal Zustand）集中列表與摘要；需求即取，不過度快取。
- 漸進增強：功能旗標驅動延遲載入可選模組。
- 快速失敗：錯誤集中於統一錯誤抽屜，不阻塞其它面板。

## 3. 資料流程與序列圖
### 3.1 啟動處理工作
```
使用者 → UI: 點擊 "Start Processing"
UI → ingestion-svc: GET /documents/:id 確認狀態
UI → processing-svc: POST /process-jobs { document_id, profile_hash }
processing-svc → UI: 202 { job_id }
UI → processing-svc: (每 10s) GET /process-jobs/:job_id
循環至終態
processing-svc → UI: { status: completed, chunk_count, embedding_profile_hash }
UI: 更新列並提示
```

### 3.2 KG 視覺化（旗標路徑）
```
UI 啟動 → FeatureFlagsProvider: GET /config/feature-flags
[kgVisualization = true]
使用者選擇 KG 分頁
UI → kg-builder-svc: GET /kg/:id/summary (node_count, relationship_count, degree_histogram, top_entities[])
UI (lazy import): import('cytoscape') → 繪製 CytoscapeGraph
```

### 3.3 評估執行監控
```
使用者觸發（或外部）
UI → eval-runner-svc: GET /eval-runs?filters
UI ↔ eval-runner-svc: 輪詢 /eval-runs/:id/stream (或 WS) 進度事件
完成後：抓取 insights adapter artifacts 更新細節面板
```

### 3.4 多執行比較（旗標）
```
使用者開啟 compare 模式 (multiRunCompare=true)
UI: 啟用 run 列選取
UI: 載入各 run kpis.json 計算差異
UI: 顯示差異徽章與趨勢火花線
```

## 4. 外部介面（消費之服務端點）
| 面板          | 服務                 | Endpoint              | 方法 | 請求                       | 回應重點                                                          |
|---------------|----------------------|-----------------------|------|----------------------------|-------------------------------------------------------------------|
| Documents     | ingestion-svc        | /documents            | GET  | n/a                        | [{km_id,version,checksum,status}]                                 |
| Documents     | ingestion-svc        | /documents/:id        | GET  | n/a                        | {status,size,hash}                                                |
| Processing    | processing-svc       | /process-jobs         | POST | {document_id,profile_hash} | 202 {job_id}                                                      |
| Processing    | processing-svc       | /process-jobs/:id     | GET  | n/a                        | {status,progress,chunk_count}                                     |
| KG            | kg-builder-svc       | /kg                   | GET  | n/a                        | [{kg_id,node_count,relationship_count,status}]                    |
| KG            | kg-builder-svc       | /kg/:id/summary       | GET  | n/a                        | {node_count,relationship_count,degree_histogram[],top_entities[]} |
| Testsets      | testset-gen-svc      | /testset-jobs         | POST | {...config}                | 202 {job_id}                                                      |
| Testsets      | testset-gen-svc      | /testset-jobs/:id     | GET  | n/a                        | {status,sample_count,seed,config_hash}                            |
| Evaluations   | eval-runner-svc      | /eval-runs            | GET  | filters                    | [{run_id,metrics_version,progress,verdict}]                       |
| Evaluations   | eval-runner-svc      | /eval-runs/:id        | GET  | n/a                        | {progress,verdict,error_count}                                    |
| Evaluations   | eval-runner-svc      | /eval-runs/:id/stream | WS   | n/a                        | 進度事件                                                          |
| Reports       | reporting-svc        | /reports              | GET  | filters                    | [{report_id,run_id,status,type}]                                  |
| Reports       | reporting-svc        | /reports/:id          | GET  | n/a                        | {status,html_url,pdf_url}                                         |
| KM Summaries  | insights-adapter-svc | /exports              | GET  | type=km_summary            | 摘要陣列                                                          |
| Feature Flags | config endpoint      | /config/feature-flags | GET  | n/a                        | 旗標布林與中繼                                                    |

## 5. 前端資料模型 (TypeScript)
```ts
export interface DocumentRow { km_id: string; version: string; checksum: string; status: string; size?: number; last_event_ts?: string }
export interface ProcessingJob { job_id: string; document_id: string; status: string; progress: number; chunk_count?: number; embedding_profile_hash?: string }
export interface KnowledgeGraphSummary { kg_id: string; node_count: number; relationship_count: number; degree_histogram?: number[]; top_entities?: { entity: string; degree: number }[] }
export interface TestsetJob { job_id: string; status: string; sample_count?: number; seed?: number; config_hash?: string }
export interface EvaluationRun { run_id: string; testset_id: string; metrics_version: string; progress: number; verdict?: string; error_count: number }
export interface ReportEntry { report_id: string; run_id: string; status: string; type: 'html'|'pdf'|'both'; html_url?: string; pdf_url?: string }
export interface KMSummary { kind: 'testset'|'kg'; schema_version: string; created_at: string; counts: Record<string, number> }
```

## 6. 狀態管理策略
- 共用 store 新增：documents[], processingJobs[], kgSummaries[], testsetJobs[], evalRuns[], reports[], kmSummaries[]。
- 派生選擇器：activeProcessing(document_id), lastKGSummaries(limit), runDelta(baseRunId, compareRunId)。
- 正規化：內部以 Map keyed by id；選擇器輸出排序陣列。

## 7. 功能旗標整合
- 啟動單次 fetch `/config/feature-flags`；Phase 1 不輪詢。
- Lazy 邊界：
  - KG 視覺化：使用者開啟分頁且 flag=true 時動態 import('cytoscape')。
  - 多執行比較：未引入額外大型圖表庫前僅啟用邏輯；未來擴大時延遲載入。
- Phase 2：支援 `refreshIntervalSeconds` 週期 re-fetch。

## 8. 錯誤處理與韌性
| 層級     | 模式                                       | 範例                                |
|----------|--------------------------------------------|-------------------------------------|
| Fetch    | 統一封裝加入 trace_id（未來）與大小限制      | fetchJson(url,{maxBytes:2_000_000}) |
| UI       | 錯誤抽屜集中 {ts,service,code,trace_id}    | pushError(err)                      |
| Retry    | Poll 5xx 指數退避 1/2/4/4                  | pollWithBackoff                     |
| Fallback | 缺少 degree_histogram 僅隱藏圖不隱藏整面板 | 條件渲染                            |
| Timeout  | AbortController 預設 10s                   | fetch signal                        |

## 9. 效能與 Bundle
- 初始 bundle 目標 ≤1.2MB gzip。
- 代碼分割：KG 視覺化 & compare 模組延遲載入。
- 清單虛擬化：>500 列時 Phase 2 採用 react-window。
- 輪詢節流：統一排程確保 ≤1 request/5s/列表 (UI-FR-050)。
- KPI 差異計算：以 run_id pair 快取鍵簡單記憶。

## 10. 國際化
- 复用既有 i18n loader；新增 lifecycle namespace。
- 缺 key 顯示 key 並記錄 (UI-FR-048)。

## 11. 無障礙
- 颜色/對比沿用現有 token。
- KG 圖出現時提供文字摘要（節點/關係數與 top entities）。
- 鍵盤順序依分頁；圖互動鍵盤支援 Phase 2 評估。

## 12. 測試策略
| 層級         | 測試                                                       |
|--------------|------------------------------------------------------------|
| 單元         | hooks（輪詢、差異計算）、旗標 provider fallback、store reducers |
| 元件         | KG 摘要面板（旗標 on/off）、Processing SLA 標記               |
| 整合         | 多執行比較選取 + 差異渲染、報告列表 PDF→HTML 回退           |
| 契約 (Mock)  | /kg/:id/summary 欄位映射驗證                               |
| 效能 (Build) | Bundle 大小與 chunk 存在/缺失檢查                          |
| E2E          | 分頁導航持久化、旗標啟用 KG 面板可見                        |

覆蓋率：新模組 ≥80%。

## 13. 追溯映射
- UI-FR-016/017/018 → KG 摘要與可視化（旗標）。
- UI-FR-023/024/026 → 評估列表 + 進度輪詢 & error_count。
- UI-FR-029 → 多執行比較覆蓋層（旗標）。
- UI-FR-033/034/035 → KM 摘要表格。
- UI-FR-049/050/051 → 效能策略（bundle、輪詢節流、事件延遲）。
- UI-NFR-006 → 分割與延遲載入。

## 14. 安全與隱私
- Phase 1 無 auth；預留 token 注入 stub。
- UI-FR-056：遮蔽秘密工具函式。
- UI-FR-058：`maskPII` 客端僅顯示層面；來源資料不改寫。

## 15. 日誌與遙測
- 結構化 logEvent({type,component,duration_ms,trace_id?})。
- 啟動一次性記錄旗標狀態（debug）。
- 錯誤抽屜未來可上報後端。

## 16. 擴充與外掛
- Dev 模式 window.__UI_EXTENSIONS__ 掃描 register(panel)。
- 面板容器按名稱避免衝突。
- 實驗指標可視化 iframe 沙箱（postMessage 白名單）。

## 17. 未來增強
| 領域    | 增強               | 理由              |
|---------|--------------------|-------------------|
| 旗標    | 簽章 + ETag        | 完整性 & 減少頻寬 |
| 圖      | 伺服端抽樣 + 聚類  | 大型 KG 效能      |
| 離線    | IndexedDB 佇列持久 | 網路韌性          |
| Auth    | OAuth2 + 服務 JWT  | 正式環境安全      |
| Replay  | 情境重播面板       | 偵錯重現性        |
| Metrics | 前端效能 HUD       | 性能治理          |

## 18. 未決議題
| 主題            | 狀態     | 下一步                |
|-----------------|----------|-----------------------|
| WebSocket 聚合  | 待定     | gateway 效能評估      |
| 離線佇列範圍    | 待定     | 定義冪等動作清單      |
| KG 摘要欄位演進 | 服務主導 | 若擴充加 version 欄位 |

## 19. 風險與緩解
| 風險         | 影響     | 緩解                  |
|--------------|----------|-----------------------|
| 旗標過度膨脹 | 複雜度   | 中央註冊 + lint 規則  |
| 輪詢壓力     | 後端負載 | 共用排程 + 退避       |
| KG 摘要太大  | 渲染慢   | Top-N 截斷 + bin 限制 |
| Bundle 成長  | 違反 NFR | CI 分析閾值           |

## 20. 實作階段
| Phase | 焦點                        | 交付                                                  |
|-------|-----------------------------|-------------------------------------------------------|
| 1     | 基礎生命週期表格 + 旗標整合 | Documents/Processing/KG/Testsets/Evaluations skeleton |
| 2     | 視覺化與比較                | Cytoscape lazy load、多執行比較                        |
| 3     | 進階運維 & Replay           | SLA 面板、情境重播、KG 篩選                             |
| 4     | 強化與授權                  | Auth、A11y 稽核、效能調優                               |

## 23. WebSocket Schema 與復原
### 23.1 封包結構 (JSON Schema 摘要)
```json
{
  "$id": "https://example.com/schemas/ui-event-envelope.schema.json",
  "type": "object",
  "required": ["topic","type","ts","seq","data"],
  "properties": {
    "topic": {"type": "string"},
    "type": {"type": "string"},
    "ts": {"type": "string", "format": "date-time"},
    "seq": {"type": "integer", "minimum": 0},
    "trace_id": {"type": "string"},
    "schemaVersion": {"type": "string", "default": "v1"},
    "data": {"type": "object"}
  }
}
```
前向相容：忽略未知欄位；後向：刪除需 ≥1 版本緩衝。

### 23.2 序號與缺口偵測
- `seq` 對每個 topic 單調增加；client 紀錄最後 seq。
- 缺口：`incoming.seq > last_seq + 1` → 標記 stale 待重抓。

### 23.3 復原演算法（偽碼）
```
onEvent(e):
  if gap(e.topic,e.seq): markStale(e.topic)
  buffer(e)
  if hasStale() && !resyncInFlight: startResync()

startResync():
  staleTopics -> REST 抓最新 snapshot
  merge snapshot; 丟棄舊緩衝
  clear stale
```

### 23.4 關閉碼（規劃）
| Code | 意義         | Client 行為              |
|------|--------------|--------------------------|
| 4000 | Auth 過期    | 重新驗證後重連 (Phase 2) |
| 4001 | 協定版本不符 | 回退輪詢並記錄錯誤       |
| 4002 | 訂閱過多     | 減少主題後重試           |
| 4003 | Rate limit   | 退避 ≥10s 再試           |

### 23.5 尺寸與壓縮
- 最大 frame: 64KB；超出丟棄並記錄 `ui.ws.frame_oversize`。
- 壓縮：permessage-deflate 選用；RTT <50ms 且 p95 <1KB 可停用。

### 23.6 心跳與逾時
- 心跳 15s；>30s 未見心跳或任何資料則重連。
- 漂移度量：實際到達 - 預期到達時間。

### 23.7 回退流程
```
poll → 嘗試 WS → (失敗) 指數重試 1/2/5/10s → 三次失敗 → 停留輪詢 2 分鐘 → 再次升級嘗試
```

## 24. KgGraph 測試與效能預算
### 24.1 單元測試矩陣
| ID        | 情境         | 預期                               |
|-----------|--------------|------------------------------------|
| KG-UT-001 | 僅摘要       | 直方圖+entities 顯示；無動態 import |
| KG-UT-002 | 啟用成功     | 流程 summary→loading→graph         |
| KG-UT-003 | 啟用失敗     | 顯示錯誤 + Retry                   |
| KG-UT-004 | 節點 600     | 顯示截斷 (500) 標記                |
| KG-UT-005 | 無直方圖     | 顯示無資料文字                     |
| KG-UT-006 | 動態載入失敗 | 降級模式 + Retry                   |
| KG-UT-007 | A11y 摘要    | 存在可讀文字區域                   |

### 24.2 整合 / E2E
| ID         | 情境            | 斷言                         |
|------------|-----------------|------------------------------|
| KG-E2E-001 | 旗標關閉        | 顯示 disabled 訊息 無 fetch  |
| KG-E2E-002 | 旗標開啟 + 啟用 | 出現 graph canvas chunk 載入 |
| KG-E2E-003 | 大型資料        | 出現 sampling pill           |

### 24.3 效能預算
| 指標                             | 預算    | 測量方式             |
|----------------------------------|---------|----------------------|
| Cytoscape chunk (gz)             | ≤300KB  | build stats          |
| KgGraph wrapper chunk            | ≤40KB   | build stats          |
| 摘要渲染時間                     | <50ms   | performance.now 測試 |
| Lazy graph 首次繪製 (≤500 nodes) | <1200ms | E2E trace            |
| 記憶體增量 (500 nodes)           | <30MB   | Heap snapshot        |

### 24.4 回歸防護
- CI 解析 Vite manifest；超過閾值（+5%）失敗。
- 合成基準腳本反覆掛載 20 次檢查記憶體洩漏 (<10% 增長)。

## 25. Telemetry 與指標規格
### 25.1 事件分類
| 類型             | 欄位                             | 觸發         |
|------------------|----------------------------------|--------------|
| ui.ws.connect    | { ts, attempt, success }         | WS 開關      |
| ui.ws.gap        | { topic,last_seq,incoming_seq }  | 缺口偵測     |
| ui.ws.resync     | { topics[],duration_ms }         | 重抓完成     |
| ui.ws.downgrade  | { reason }                       | 降級輪詢     |
| ui.kg.enable     | { node_goal,fetch_ms }           | 啟用 KG      |
| ui.kg.render     | { nodes,edges,mode,duration_ms } | 初次版面完成 |
| ui.flag.snapshot | { flags,schemaVersion }          | 啟動         |

### 25.2 衍生指標
- WS Uptime% = 連線時間 / Session 時間。
- Reconnect Rate = 每小時重連事件數。
- KG Layout p95 時間。
- Flag Fetch 失敗率。

### 25.3 日誌與取樣
- 高頻 progress.update 不逐條記錄；彙總累加並 60s flush。
- 錯誤類必記。

### 25.4 隱私
- 不含 PII；符合秘密樣式字串遮罩。

## 26. ADR 參考 (規劃)
| ADR ID  | 標題                 | 狀態     | 摘要                           |
|---------|----------------------|----------|--------------------------------|
| ADR-001 | KG 視覺化套件選擇    | Accepted | Cytoscape.js + Lazy + 摘要備援 |
| ADR-002 | 單一多工 WS 通道     | Proposed | 降低多連線成本簡化授權         |
| ADR-003 | 功能旗標合併策略     | Draft    | Client 預設覆蓋 + 忽略未知     |
| ADR-004 | Graph 抽樣上限 (500) | Draft    | 控制初始渲染時間與記憶體       |

未來：每份 ADR 將放置 `docs/adr/ADR-xxx-*.md`，含背景/決策/影響。

## 27. Subgraph API 草稿（探索性）
目的：提供按需聚焦子圖，支援未來 UI 篩選 / 上下文擴展，而不需下載完整 KG。

### 27.1 Endpoint
`GET /kg/{kg_id}/subgraph`

### 27.2 查詢參數
| 參數           | 類型       | 必填                | 說明               | 限制               |
|----------------|------------|---------------------|--------------------|--------------------|
| center         | string     | 是（除非使用 entity） | 節點 ID 或實體標籤 | 不存在則 404       |
| entity         | string     | 與 center 擇一      | 精確實體字串       | 與 center 互斥     |
| hop            | int        | 選填                | 無向 hop 距離      | 1..3 (預設 1)      |
| max_nodes      | int        | 選填                | 節點上限           | 50..500 (預設 200) |
| relation_types | csv string | 選填                | 關係 kind 過濾     | 白名單驗證         |
| degree_min     | int        | 選填                | 最小度數           | >=0                |
| degree_max     | int        | 選填                | 最大度數           | >= degree_min      |
| summarize      | bool       | 選填                | 是否包含聚合統計   | 預設 true          |
| version        | string     | 選填                | KG Schema 版本     | 預設現行           |

### 27.3 回應（schemaVersion v1）
```json
{
  "kg_id": "uuid",
  "center": "node-123",
  "hop": 2,
  "node_count": 180,
  "edge_count": 340,
  "truncated": true,
  "nodes": [
    { "id": "node-123", "label": "鋼板", "degree": 42, "entityType": "material" },
    { "id": "node-987", "label": "檢查", "degree": 12 }
  ],
  "edges": [
    { "id": "e-1", "source": "node-123", "target": "node-987", "kind": "action", "weight": 0.76 }
  ],
  "summary": {
    "degree_histogram": [5,12,40,22,7],
    "top_entities": [{"entity":"鋼板","degree":42},{"entity":"檢查","degree":12}],
    "avg_weight": 0.55,
    "density": 0.021
  },
  "schemaVersion": "v1"
}
```

### 27.4 截斷與抽樣
- 若結果節點數 > `max_nodes`，以 deterministic sampling（hash(id) %）確保相同條件重現。
- `truncated=true` 指示抽樣；`node_count`/`edge_count` 仍保留未抽樣原始總數。

### 27.5 錯誤模型
| HTTP | Code             | 訊息示例                                   | 備註       |
|------|------------------|--------------------------------------------|------------|
| 400  | VALIDATION_ERROR | "max_nodes out of range (min 50, max 500)" | 參數限制   |
| 400  | MUTUAL_EXCLUSION | "center and entity are mutually exclusive" | 擇一       |
| 404  | NOT_FOUND        | "center node not found"                    | 節點不存在 |
| 410  | GONE             | "kg version deprecated"                    | 版本停用   |
| 429  | RATE_LIMIT       | "too many subgraph requests"               | 防濫用     |

### 27.6 快取與 ETag
- Key: (kg_id, center|entity, hop, filters, max_nodes, version)。
- 推薦：`Cache-Control: public, max-age=30, stale-while-revalidate=60`。
- ETag 基於排序後 nodes+edges；排除易變欄位。

### 27.7 Rate Limiting 建議
- Soft：每 user 每 kg_id 每分鐘 30 次。
- Burst：5 次 / 5 秒 觸發 429（含 Retry-After）。

### 27.8 安全 / 濫用考量
- 抽樣前先套用 PII 遮罩避免洩漏。
- hop > 3 一律拒絕防止爆炸式展開。
- 監控 p95 payload；若 >1MB 調整 max_nodes 預設。

### 27.9 未來擴充
- `direction=out|in|both` 有向過濾。
- 加權半徑（累計邊權重門檻停止）。
- 伺服端 layout hints `{positions:{id:{x,y}}}` 降低客端計算。

### 27.10 開放問題
- 抽樣是否需按 entityType 分層保持少數類別代表性？
- 是否加入來源 chunk 極簡 provenance（每節點前 2 個）？
- 需不需要 POST /kg/subgraphs 支援多 center 批次？
 
## 27.11 ADR 參考
| ADR ID  | 標題 (中文近似)                  | 狀態  | 關聯節   | 影響摘要                 |
|---------|----------------------------------|-------|----------|--------------------------|
| ADR-001 | 微服務結構                       | Draft | §2       | UI 端端點與服務邊界對應  |
| ADR-002 | 知識圖譜視覺化技術（Cytoscape.js） | Draft | §22      | Lazy 載入 + 圖渲染策略   |
| ADR-003 | Subgraph 採樣策略                | Draft | §22, §27 | 決定性抽樣 + 尺度上限    |
| ADR-004 | Manifest 完整性與追溯            | Draft | §3, §25  | 日後完整性檢查 UI 鉤子   |
| ADR-005 | Telemetry 分類與命名             | Draft | §15, §25 | 事件/指標命名一致性      |
| ADR-006 | 事件 Schema 版本策略             | Draft | §21, §23 | WebSocket 與事件演進治理 |

未來新增 ADR（例如 Feature Flag 合併策略）將加入此表並標註影響面。
## 21. WebSocket 事件介面（草稿）
### 21.1 通道策略
初期仍可維持輪詢；此處規劃未來升級為單一多工 WebSocket：`wss://<gateway>/ui/events`。客戶端以 JSON 握手訂閱主題。

握手：
```json
{ "action":"subscribe", "topics":["documents","processing","kg","testsets","eval_runs","reports"], "client_version":"ui-0.1", "trace_id":"<uuid>" }
```

伺服端事件封包：
```json
{
  "topic": "eval_runs",
  "type": "progress.update",
  "ts": "2025-09-10T10:00:12.345Z",
  "trace_id": "...",
  "data": { "run_id": "uuid", "progress": 42, "error_count": 1, "partial_metrics": { "faithfulness": 0.81 } }
}
```

主題與型別 (MVP)：
| Topic      | Types                            | Data                                             |
|------------|----------------------------------|--------------------------------------------------|
| documents  | status.update                    | { document_id,status }                           |
| processing | progress.update                  | { job_id,progress,chunk_count? }                 |
| kg         | build.update                     | { kg_id,status,node_count?,relationship_count? } |
| testsets   | job.update                       | { job_id,status,sample_count? }                  |
| eval_runs  | progress.update,completed,failed | { run_id,progress,error_count?,verdict? }        |
| reports    | status.update                    | { report_id,status }                             |

### 21.2 客戶端處理
- 單一連線 hook：`useEventStream()` 指數回連 (1s→2s→5s→10s 上限) + 抖動。
- 事件派發表路由至 store mutators；未知型別僅 debug 記錄。
- 回壓：訊息佇列 > 500 時切入降級模式（暫停處理，透過 REST 重整後恢復）。

### 21.3 重整協議
若偵測心跳遺失 (>30s) 或序號跳躍，發出並行 REST 重新抓取受影響主題，完成後恢復增量。

Heartbeat：
```json
{ "topic":"control", "type":"heartbeat", "ts":"2025-09-10T10:00:15Z" }
```
逾時：2 * 心跳間隔（預設 15s）。

### 21.4 安全 / 未來
- Phase 2：握手附 JWT 以授權主題。
- 可選每幀 HMAC 簽章（旗標影響敏感視覺時）。

## 22. KG 視覺化組件草稿
### 22.1 職責
`KgGraph` 呈現知識圖譜的抽樣或摘要（UI-FR-018），不提供編輯；優先快速載入、低 bundle 影響與優雅退化。

### 22.2 Props 合約
```ts
interface KgGraphProps {
  summary: KnowledgeGraphSummary
  fetchFullGraph?: () => Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }>
  height?: number
  theme?: 'light' | 'dark'
}
interface GraphNode { id: string; label: string; degree?: number; entityType?: string }
interface GraphEdge { id: string; source: string; target: string; kind?: string; weight?: number }
```

### 22.3 呈現模式
| 模式        | 觸發                   | 行為                                            |
|-------------|------------------------|-------------------------------------------------|
| SummaryOnly | 預設                   | 顯示度數直方圖、Top entities、"Enable Graph" 按鈕 |
| LazyGraph   | 使用者啟用且 flag=true | 動態 import cytoscape 建構元素                  |
| Expanded    | fetchFullGraph 完成    | 使用 cose-bilkent 版面並支援 fit                |
| Degraded    | 載入/版面錯誤          | 顯示文字摘要備援                                |

### 22.4 動態載入模式
```ts
const CytoscapeGraph = React.lazy(() => import(/* webpackChunkName: "kg-viz" */ './CytoscapeGraph'))
```
Suspense fallback：骨架 + 文字摘要。

### 22.5 版面與樣式
- ≤200 節點 concentric；>200 切換 cose-bilkent（必要時延遲載入外掛）。
- 節點顏色：度數百分位分群（五分位）。邊透明度依 weight。
- 暗色模式以 CSS 變數切換背景與描邊對比。

### 22.6 效能考量
- 初始節點上限 500；超出加 "Sampled 500 of N" 提示。
- >800 節點可用 WebGL（可用時）；否則 canvas。
- Resize / fit 100ms debounce。

### 22.7 無障礙
- ARIA live 區域宣告："Graph loaded with X nodes and Y edges"（可本地化）。
- 提供表格切換列出 top entities。

### 22.8 錯誤案例
| 情境                | 處理                                     |
|---------------------|------------------------------------------|
| 動態 import 失敗    | 記錄錯誤 + Retry 按鈕                    |
| fetchFullGraph 拒絕 | 維持摘要 + "Full graph unavailable" 提示 |
| >2MB JSON           | 中止請求 → 摘要模式                      |

### 22.9 測試掛鉤
- data-testids: kg-summary-hist, kg-top-entities, kg-enable-btn, kg-graph-canvas。
- 單元測試以 mock 動態 import 取代真實 cytoscape。

### 22.10 未來增強
- 子圖篩選（entity type / 度數範圍）。
- 節點詳細面板（來源 chunk 參考）。
- Snapshot 匯出（PNG/SVG）。

---
文件結束。

---

## 28. 端到端資料流觀點 (End-to-End Data Flow Perspective)

以下為整合（requirements.md, requirements.ui.md, design.md, design.ui.md）之「端到端資料流(Data Flow)」視角，補強第 3 節序列圖與第 13 節追溯，覆蓋：輸入來源 → 轉換處理 → 工件(artifacts) → 事件(events) → UI/Portal/KM 消費與追溯與風險防護。  

---

### 28.1 高階層次視圖
層級 (上到下)  
1. 使用者 / 角色：KM 工程師、Processing 工程師、KG 構建工程師、Testset 生成工程師、RAG 評估工程師、PM/分析師。  
2. UI 模組（Lifecycle Console + 現有 Insights Portal）：Documents / Processing / KG / Testsets / Evaluations / Insights / Reports / KM Summaries / Admin。  
3. 服務 API：ingestion-svc → processing-svc → (optional) kg-builder-svc → testset-gen-svc → eval-runner-svc → insights-adapter-svc → reporting-svc → config/feature-flags endpoint。  
4. 儲存與工件層：Object Storage (raw docs, chunks, testsets, evaluation_items.json, kpis.json, personas.json, export_summary.json, report.html/pdf, run_meta.json)、Vector Store(pgvector)、KG JSON、KM export summaries (`testset_summary_v0`, `kg_summary_v0`)。  
5. 事件/更新通道：輪詢 + （未來）WebSocket (progress.update, status.update, completed 等)。  
6. 外部系統：KM REST（讀→文件內容；Phase 後續：寫→摘要）、Portal 視覺化。  

---

### 28.2 階段式資料流總覽表

| 階段                   | 主要輸入                            | 轉換邏輯                                                        | 主要輸出工件 / 資料                                                        | 事件                                      | 下游消費者                 | 需求/設計參照                                                                       |
|------------------------|-------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------|-------------------------------------------|----------------------------|-------------------------------------------------------------------------------------|
| Ingestion              | KM 文件引用 (km_id, version) 或本地 | 下載、checksum、去重                                              | raw doc 檔案、documents 列表記錄                                            | document.ingested                         | Processing 面板            | requirements.md FR（文件匯入段）；design.md 2.x / 4.x；requirements.ui.md UI-FR-003/004 |
| Processing             | document_id                         | 內容抽取、清洗、chunking、embedding                                | chunks.jsonl、向量、processing job 狀態                                      | document.processed / processing.completed | Testset、KG                 | design 3.x；UI-FR-008~012                                                            |
| Knowledge Graph (可選) | chunks (與其 metadata)              | 實體/關鍵詞抽出、關係構建 (Jaccard/Overlap/Cosine/SummaryCosine) | graph.json (nodes, relationships)、kg summary (node/rel counts, histograms) | kg.built                                  | Testset(策略)、UI KG 分頁   | design 3.3 / 20；UI-FR-016~018                                                       |
| Testset Generation     | chunks(+KG) + 策略/seed             | 問題/答案合成、情境/Persona、去重                                 | samples.jsonl、personas.json、scenarios.json、testset meta                    | testset.created                           | Evaluation                 | requirements FR-013~016；UI-FR-019~022                                               |
| Evaluation Run         | testset_id + RAG Profile            | 查詢 RAG 系統 → 收集上下文 → 計算指標(RAGAS擴充)                | evaluation_items.json、kpis.json、thresholds.json(可選)                      | eval_runs.progress.update / run.completed | Insights Adapter、Reporting | requirements FR-017~022；UI-FR-023~026                                               |
| Insights Adapter       | evaluation artifacts                | 正規化、聚合、可選匯總 (kg/persona stats)                         | export_summary.json (可選)                                                 | adapter.exported                          | Insights Portal 視圖       | FR-038/039；UI-FR-027~029                                                            |
| Reporting              | run artifacts                       | 模板渲染 HTML / PDF                                             | report.html / report.pdf、run_meta.json 更新 URL                            | report.completed                          | UI Reports 分頁            | FR-026/027, FR-037~040；UI-FR-030~032                                                |
| KM Export (初始)       | testset & kg summary 資料           | 過濾 & 萃取摘要 (不含原文/敏感)                                 | testset_summary_v0.json、kg_summary_v0.json                                 | km.exported                               | KM 系統, UI KM Summaries   | FR-041/042；UI-FR-033~035                                                            |
| Feature Flags          | 靜態/服務設定                       | 旗標合併、一次載入                                               | feature-flags snapshot (client memory)                                     | (無)                                      | 所有延遲載入模組           | UI-NFR-006；design.ui.md 7                                                           |
| Subgraph API (草稿)    | kg_id + center/entity               | 節點截斷 / 抽樣 / 統計                                          | subgraph JSON                                                              | (未定)                                    | KG 可視化互動              | design.ui.md §27                                                                    |

---

### 28.3 事件序列（代表性路徑）

### 3.1 完整執行（含 KG）
1. UI Documents 面板 → GET /documents  
2. 使用者觸發處理 → POST /process-jobs  
3. 輪詢 /process-jobs/:id 直到 completed  
4. 若啟用 KG → POST/GET /kg (建構) → GET /kg/:id/summary  
5. 生成測試集 → POST /testset-jobs → 輪詢 /testset-jobs/:id  
6. 啟動評估 → POST /eval-runs → 進度（輪詢或未來 WS progress.update）  
7. run.completed 後 → 抓 evaluation_items.json / kpis.json → insights adapter export_summary.json（可選）  
8. 生成報告 → GET /reports?run_id=… → run_meta.json 更新 report_html_url / report_pdf_url  
9. KM 初始匯出（自動或排程）→ testset_summary_v0 / kg_summary_v0 → UI KM Summaries 顯示  

(參照：design.md §4 Data Flow, §18 KM Storage；design.ui.md §3 Sequence；requirements.ui.md 各 5.x 功能段)

### 3.2 評估進度即時（未來 WS）
- UI 建立單一 WebSocket (topic: eval_runs, reports)  
- 收到 progress.update → 更新 evalRuns[] store 中 progress / error_count  
- 序號缺口觸發快照重新擷取 (design.ui.md §23 / §21)  

---

### 28.4 工件(Artifacts) 血緣鏈 (Lineage Chain)
document(raw) → chunks → (optional graph.json) → test samples (samples.jsonl) → evaluation_items.json → kpis.json → export_summary.json (可選) → run_meta.json (含報告 URL) → KM summaries (最小化)  

對應追溯目標：requirements 目標 #4 (≥95% coverage)。  
支援 UI Traceability（UI-FR-060+ audit 類）與 design.ui.md §13 追溯映射。  

---

### 28.5 Idempotency / 決定性點
| 階段       | 決定性輸入雜湊                        | 作用                        |
|------------|---------------------------------------|-----------------------------|
| Processing | (document_id + profile_hash)          | 避免重複切片/embedding      |
| KG Build   | (kg_build_config hash)                | 重建 vs 重用                |
| Testset    | (seed + config hash + optional kg_id) | 重產同問題集                |
| Evaluation | (testset_id + rag_profile_hash)       | 相同組合可落入 cache (未來) |
| KM Export  | (resource_type + source_run_id)       | 避免重複上傳摘要            |

---

### 28.6 主要資料模型（摘要）
| 模型                   | 關鍵欄位                                                           | 來源                     | 消費者                    |
|------------------------|--------------------------------------------------------------------|--------------------------|---------------------------|
| DocumentRow            | km_id, version, checksum, status                                   | ingestion-svc            | UI Documents              |
| ProcessingJob          | job_id, progress, chunk_count                                      | processing-svc           | UI Processing             |
| KnowledgeGraphSummary  | node_count, relationship_count, degree_histogram[], top_entities[] | kg-builder-svc           | UI KG / KM summary        |
| TestsetJob             | job_id, sample_count, config_hash                                  | testset-gen-svc          | UI Testsets               |
| EvaluationRun          | run_id, progress, verdict, error_count                             | eval-runner-svc + events | UI Evaluations / Insights |
| ReportEntry            | report_id, run_id, status, html_url/pdf_url                        | reporting-svc + run_meta | UI Reports                |
| KMSummary (testset/kg) | schema_version, counts{}                                           | adapter / export process | UI KM Summaries / KM DB   |
| Subgraph (草稿)        | nodes[], edges[], truncated, summary                               | subgraph endpoint        | KG 可視化                 |

(對應定義散見於 design.ui.md §5、design.md §5、KG 延伸 §20 / Subgraph §27)

---

### 28.7 UI 資料提取策略
| 面板          | 初次載入策略                 | 更新策略        | 回退                           |
|---------------|------------------------------|-----------------|--------------------------------|
| Documents     | 即時 GET /documents          | 每 10s 輪詢     | 失敗 → 指數退避                |
| Processing    | POST 後輪詢 status           | 10s → completed | 連續失敗 >3 → 標記 degraded    |
| KG            | GET /kg/:id/summary          | 手動 Refresh    | 若圖庫 disabled → 摘要         |
| Testsets      | POST job → 輪詢              | 10s             | 失敗 → 顯示 last_error         |
| Evaluations   | GET /eval-runs               | 輪詢或 WS       | WS 降級 → 輪詢                 |
| Reports       | GET /reports?run_id          | 30s 或事件      | PDF 404 → HTML fallback        |
| KM Summaries  | GET /exports?type=km_summary | 手動/刷新按鈕   | Schema 驗證失敗 → invalid 標籤 |
| Feature Flags | GET /config/feature-flags    | Phase1 單次     | 失敗 → 預設旗標                |

(映射 UI-FR-050 輪詢節流、UI-FR-031 報告狀態、UI-FR-034 KM diff)

---

### 28.8 事件 / API 與 UI-FR 對照

| UI-FR         | 事件 / API                            | 敘述                |
|---------------|---------------------------------------|---------------------|
| UI-FR-023/024 | GET /eval-runs + (WS progress.update) | 進度即時更新        |
| UI-FR-030~032 | report.completed + GET /reports/:id   | 報告列表與 fallback |
| UI-FR-033~035 | GET /exports?type=km_summary          | KM 摘要刷新與驗證   |
| UI-FR-016~018 | GET /kg/:id/summary (+ Subgraph 草稿) | KG 摘要 + 視覺化    |
| UI-FR-049~051 | 節流輪詢 + WS 延遲處理                | 效能 SLA            |
| UI-FR-053~055 | 統一錯誤抽屜 + 離線佇列策略           | 錯誤/可靠性         |

---

### 28.9 KM 初期匯出資料流 (最小化)
1. 選擇性流程：Run / KG 建立完成 → 內部導出程式取計數  
2. 生成：`testset_summary_v0.json` / `kg_summary_v0.json` (schema_version, counts)  
3. 儲存：`km_exports/...` + 寫入 export index (未來)  
4. UI KM Summaries 面板拉取顯示；KM 系統稍後以只讀 API 抓取或檔案同步  
5. 不含：問題文字、chunk 原文、embedding（滿足最小暴露原則）  

(參照 requirements.md FR-041/042；design.md §18 KM Storage；UI 對應 UI-FR-033~035)

---

### 28.10 Subgraph 流 (草稿)
UI 於 KG 分頁提供「Focus Entity」介面：  
1. 使用者輸入 entity / 選 node → GET /kg/{kg_id}/subgraph?entity=...&hop=2&max_nodes=200  
2. 後端執行 BFS/鄰接抽樣 → 產出截斷子圖 (truncated flag)  
3. 前端 Cytoscape lazy chunk 載入渲染；若 truncated=true 顯示 sampling pill  
4. 未來可將子圖結果快取於 local store → 滑動還原  

(參照 design.ui.md §27；中文對應 §27)

---

### 28.11 追溯鏈與 UI 映射位置
| 資料鏈節點        | UI 顯示位置                    | 互動                               |
|-------------------|--------------------------------|------------------------------------|
| document_id       | Documents 表格                 | 點擊展開 raw meta                  |
| chunk_count       | Processing Jobs 行             | Hover 顯示進度                     |
| kg_id             | KG 摘要 header                 | 查看節點/關係數                    |
| testset_id        | Testsets Job 行                | Drill-down→ Persona/Scenario count |
| run_id            | Evaluations 行 + Insights 視圖 | Compare / Insights 展示指標        |
| report_*_url      | Reports 行                     | 下載 / 預覽                        |
| export summary id | KM Summaries 行                | Diff 高亮                          |
| subgraph request  | KG 可視化面板                  | Node focus / sampling              |

---

### 28.12 風險 & 資料流保護點
| 風險              | 層        | 緩解                                  |
|-------------------|-----------|---------------------------------------|
| 輪詢壓力          | API       | 全域節流 (UI-FR-050)                  |
| 大型 KG JSON 過大 | KG        | 摘要 + Subgraph 抽樣                  |
| 評估進度落差      | Eval Runs | WS 缺口偵測 + REST Resync             |
| 報告 URL 失效     | Reports   | HTML fallback + run_meta 再取         |
| 匯出摘要不一致    | KM Export | schema_version + hash (未來 manifest) |
| 子圖爆炸          | Subgraph  | hop ≤3 + max_nodes ≤500               |

---

### 28.13 與現有 Insights Portal 整合邊界
- 共用：i18n、Zustand store、指標視圖（Evaluation 完成後跳轉 Insights）。  
- 新增：Lifecycle 專屬 store slice + KG 視覺化 lazy chunk。  
- 雙模式：Artifacts 分析 (Portal 原有) + Pipeline 控制 (Lifecycle 新增)。  
- Feature Flags：Portal 原 flags 擴展加入 lifecycleConsole, kgVisualization, multiRunCompare 等（design.ui.md §7）。  

---

### 28.14 後續演進鉤子
| 項目                         | 當前狀態              | 下一步                              |
|------------------------------|-----------------------|-------------------------------------|
| WebSocket                    | 規格完成 (設計)       | Gateway / auth P2                   |
| Derivatives 回寫 KM          | Deferred (DR-001/002) | 決策後擴增 export adapter           |
| Subgraph API                 | 草稿 (Spec)           | 實驗 endpoint + Rate 測試           |
| Persona/Scenario UI 深度視圖 | 基礎列顯示            | Persona drill-down + scenario trace |
| 事件 Schema Registry         | 未實作                | JSON Schema 驗證 + 版本控管         |

---

### 28.15 對文件條目之交叉引用 (摘選)
| 文件               | 章節                                           | 本回答關聯                                   |
|--------------------|------------------------------------------------|----------------------------------------------|
| requirements.md    | Vision / FR-037~042 / KM Export / Derivatives  | 定義報告 URL、KM 摘要輸出                     |
| design.md          | §4 Data Flow / §18 KM Storage / §20 KG Library | 資料流、儲存策略、KG 技術選擇                  |
| requirements.ui.md | UI-FR-016~035 / 049~055                        | KG / Evaluations / KM Summaries / 性能與錯誤 |
| design.ui.md       | §3 序列圖 / §5 模型 / §23~27                   | UI 端資料模型 / WS / Subgraph API 草稿       |

---

### 28.16 建議（若要加強資料流健壯性）
1. 增加 run_meta.json 中 testset_id, kg_id 直接欄位（簡化 UI Trace）。  
2. 導出 manifest.json（列出所有 artifact 名稱 + sha256）→ Portal 開啟時做完整性檢查。  
3. 引入事件 Envelope JSON Schema 驗證（client drop 不合規事件並計數）。  
4. KM summary 增加 `source_run_ids[]` 以便後續多 run 聚合。  
5. Subgraph API 回應中回傳 `sampling_method`（deterministic_hash / top_degree）提升可解釋性。  
