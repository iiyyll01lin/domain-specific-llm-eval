# UI 平台需求（評估工作流程主控台）

版本：0.1（草稿）
狀態：徵求審閱
日期：2025-09-10
負責：平台工程組

---
## 1. 目的
提供統一 Web 介面以操作、監控並分析多階段 RAG 評估流程（文件匯入 → 前處理 → KG 建構 → 測試集生成 → 評估 → 洞察/報告），取代零散 CLI 使用，並支援角色導向視圖。

## 2. 範圍與非目標
範圍：
- 透過服務 API 觸發與監看工作階段（既有與未來微服務）。
- 工程/分析角色儀表板。
- 基本執行工件瀏覽與下載。
- 呈現 KM 摘要（testset 與 KG；對應 FR‑041/042）。
- 容器化部署（開發 docker-compose；生產可對接 k8s）。
非目標（v1）：
- 模型推論 Playground。  
- 衍生資源完整管理（待 DR‑001/DR‑002）。  
- 多租戶隔離。

## 3. 角色（Persona）
- KM 工程師 (KME)
- 資料處理工程師 (DPE)
- 知識圖譜工程師 (KGE)
- 測試集生成工程師 (TGE)
- RAG 評估工程師 (REE)
- QA / 評估分析師 (QA)
- 產品 / 專案經理 (PM)
- 平台 / SRE (SRE)
- 安全與法遵 (SEC)

## 4. EARS 標註說明
- Ubiquitous：恆常行為
- Event-driven：事件/動作觸發
- State-driven：依賴狀態持久
- Optional / Feature-flagged：功能旗標啟用才生效
- Unwanted Behavior：負向情境處理

## 5. 功能性需求（UI-FR）

### 5.1 導航與工作階段
UI-FR-001 (Ubiquitous) 系統應提供依生命週期分段之全域導航：Documents, Processing, KG, Testsets, Evaluations, Insights, Reports, Admin。
UI-FR-002 (State-driven) 系統應持久化上次選取的導航分頁於本地並於重新載入時還原。
UI-FR-003 (Ubiquitous) 系統應顯示工作階段指示（persona、語系、環境、workflow ID）。
UI-FR-004 (Event-driven) 當使用者切換 persona 時，系統應即時調整可見儀表與預設篩選且不整頁重載。
UI-FR-005 (Unwanted Behavior) 若 persona 設定載入失敗，系統應回退至安全最小配置。

### 5.2 驗證與存取（Phase 2 預備）
UI-FR-006 (Ubiquitous) 系統應支援可選的驗證模式；停用時顯示開發模式橫幅。
UI-FR-007 (Event-driven) 當 Token 過期，系統應提示重新驗證且保留未儲存篩選。
UI-FR-008 (Unwanted Behavior) 若 API 回傳 403，系統應隱藏禁用動作並提示權限需求。

### 5.3 文件匯入視圖
UI-FR-009 (Ubiquitous) 系統應列出已匯入文件：km_id, version, checksum, size, status, last_event_ts。
UI-FR-010 (Event-driven) 當使用者提交新文件引用時，系統應呼叫 ingestion-svc POST /documents 並顯示追蹤。
UI-FR-011 (State-driven) 文件處理中 (processing) 時列應 ≤10 秒自動更新。
UI-FR-012 (Unwanted Behavior) 若匯入失敗 (error)，系統應提供具權限使用者重試。

### 5.4 前處理工作視圖
UI-FR-013 (Ubiquitous) 系統應顯示處理工作進度（% chunks）與 embedding profile hash。
UI-FR-014 (Event-driven) 當接收 document.ingested 事件時，系統應顯示「開始處理」快捷。
UI-FR-015 (Unwanted Behavior) 若處理時間超出 SLA，系統應標記警示並顯示已耗時。

### 5.5 知識圖譜儀表
UI-FR-016 (Ubiquitous) 系統應列出 KG 建構：node_count, relationship_count, build_profile_hash, status。
UI-FR-017 (Event-driven) 當接收 kg.built 事件，系統應更新節點與關係增量。
UI-FR-018 (Optional / Feature-flagged) 啟用 KG 視覺化旗標時顯示圖摘要（度分佈與熱門實體）。

### 5.6 測試集生成控制台
UI-FR-019 (Ubiquitous) 系統應提供表單設定：method, max_samples, seed, persona profile。
UI-FR-020 (Event-driven) 當提交 testset 工作，系統應顯示 queued → running → completed 時間線。
UI-FR-021 (Unwanted Behavior) 若 sample_count 超過上限，系統應於提交前阻擋並標記。
UI-FR-022 (State-driven) 工作建立後顯示可重現雜湊（config hash + seed）。

### 5.7 評估執行協調
UI-FR-023 (Ubiquitous) 系統應列出評估：run_id, testset_id, metrics_version, progress, verdict。
UI-FR-024 (Event-driven) 當進度更新時 (WS/輪詢) 系統應更新進度與局部指標。
UI-FR-025 (Event-driven) 當 run 完成，系統應連結 insights 輸出與報告。
UI-FR-026 (Unwanted Behavior) 若樣本指標計算失敗，系統應遞增 error_count 並提供 last_error 提示。

### 5.8 洞察整合
UI-FR-027 (Ubiquitous) 系統應嵌入或連結既有 Insights Portal 視圖並重用其元件。
UI-FR-028 (Event-driven) 當使用者於評估清單選擇 run 並開啟 Insights，系統應傳遞 run_id 並預載指標。
UI-FR-029 (Optional / Feature-flagged) 啟用多執行比較模式時允許最多 5 個 run 並標示 KPI 差異。

### 5.9 報告與工件
UI-FR-030 (Ubiquitous) 系統應列出報告並提供 HTML 預覽與 PDF 下載。
UI-FR-031 (Event-driven) 報告生成期間顯示轉動指示，直至 report.completed。
UI-FR-032 (Unwanted Behavior) 若 PDF 404，系統應退回 HTML 並標註降級。

### 5.10 KM 匯出摘要
UI-FR-033 (Ubiquitous) 系統應顯示 testset 與 KG KM 摘要（schema_version, created_at）。
UI-FR-034 (Event-driven) 新摘要出現時，系統應比較前值並高亮增量。
UI-FR-035 (Unwanted Behavior) 若 Schema 驗證失敗，系統應標記 invalid 並禁止對外發布。

### 5.11 運維可觀測
UI-FR-036 (Ubiquitous) 系統應顯示服務健康（ingestion, processing, testset-gen, eval-runner, insights-adapter, reporting）。
UI-FR-037 (Event-driven) 當健康狀態轉為不健康，系統應顯示警示橫幅與時間戳。
UI-FR-038 (State-driven) 提供診斷面板：最新 trace_id 與錯誤率。

### 5.12 環境與部署
UI-FR-039 (Ubiquitous) 系統應顯示目前 API Base URL 與 feature flags（prod 唯讀）。
UI-FR-040 (Event-driven) 編輯 dev 端點時需先 HEAD /health 驗證可達後再儲存。
UI-FR-041 (Unwanted Behavior) 若本地 Docker 服務不可達，顯示修復建議（容器名、埠）。

### 5.13 擴充與外掛
UI-FR-042 (Ubiquitous) 系統應於啟動載入可設定目錄 ESM 外掛模組（dev）。
UI-FR-043 (Event-driven) 外掛註冊失敗時記錄結構化錯誤並隔離不崩潰。
UI-FR-044 (Optional / Feature-flagged) 啟用實驗視覺化時以 iframe/worker 沙箱隔離。

### 5.14 國際化與無障礙
UI-FR-045 (Ubiquitous) 系統應支援 en-US 與 zh-TW。
UI-FR-046 (Event-driven) 切換語系時不整頁重載即時更新。
UI-FR-047 (Ubiquitous) 系統應符合 WCAG 2.1 AA 對比。
UI-FR-048 (Unwanted Behavior) 缺少翻譯 key 時記錄並顯示 key。

### 5.15 效能與回應
UI-FR-049 (Ubiquitous) 首次 Documents 載入 ≤2s（文件 ≤50）。
UI-FR-050 (State-driven) 輪詢列表刷新頻率 ≤ 每 5 秒一次/列表。
UI-FR-051 (Ubiquitous) 評估進度事件到 UI 更新 ≤300ms。
UI-FR-052 (Unwanted Behavior) API 回應 >2MB 時顯示警告建議啟用伺服端分頁。

### 5.16 可靠與錯誤處理
UI-FR-053 (Ubiquitous) 系統應提供統一錯誤抽屜（code, service, trace_id, ts）。
UI-FR-054 (Event-driven) 網路中斷進入離線模式並佇列可冪等 POST；恢復後送出。
UI-FR-055 (Unwanted Behavior) 佇列動作超過重試時限則丟棄並通知。

### 5.17 安全與隱私
UI-FR-056 (Ubiquitous) 系統應遮蔽秘密（僅末 4 碼可見）。
UI-FR-057 (Event-driven) 複製秘密時記錄稽核（使用者、雜湊）。
UI-FR-058 (Unwanted Behavior) 若回應含 PII 標記欄位，系統應遮罩。

### 5.18 追溯與稽核
UI-FR-059 (Ubiquitous) 系統應記錄建立/更新稽核事件（who, what, when, resource_ref, trace_id）。
UI-FR-060 (Event-driven) 檢視 run 時顯示血緣鏈（document → chunk → test sample → evaluation item）。
UI-FR-061 (Unwanted Behavior) 若血緣解析失敗，標記未解析節點並繼續顯示。

### 5.19 CLI 過渡
UI-FR-062 (Ubiquitous) 系統應顯示對應舊 CLI 指令視圖供參考。
UI-FR-063 (Event-driven) 新工作建立時產生 CLI 指令片段可複製。
UI-FR-064 (Unwanted Behavior) 無對應 CLI 參數時標註 (UI-only)。

### 5.20 未來衍生資源預留
UI-FR-065 (Ubiquitous) 系統應保留 Derivatives 分頁佔位（列出延後類型及 deferred 標籤）。
UI-FR-066 (Event-driven) 當 DR-001/DR-002 關閉啟用建立/列表而不需大幅改版。
UI-FR-067 (Unwanted Behavior) 缺必填草稿欄位之 derivative 應高亮並排除匯出。

## 6. 非功能（UI-NFR）
UI-NFR-001 可用性：核心導航 prod 可達性 ≥99%。
UI-NFR-002 效能：P95 分頁切換互動延遲 <500ms（快取情境）。
UI-NFR-003 國際化：新字串合併前需完成雙語檢查。
UI-NFR-004 無障礙：CI axe-core 0 個 critical。
UI-NFR-005 可觀測：前端記錄 ≥90% API 呼叫含 trace_id。
UI-NFR-006 Bundle 大小：初始 JS bundle ≤1.2MB gzip（基線 persona）。
UI-NFR-007 漸進增強：停用 JS 時可顯示最小只讀狀態（Documents 簡表）。

## 7. 開放問題
- Insights Portal 嵌入或合併？（初始決策：合併為單一延伸模組，共用元件。）
- KG 視覺化套件選擇（已解決：採用 Cytoscape.js；詳見設計文件第 20 節，將以延遲載入策略減少初始 bundle）。
- WebSocket 直連或 gateway 聚合？（趨勢：gateway 一致驗證。）
- 離線佇列動作範圍（僅冪等 create?）。
- Derivatives 分頁啟用條件（DR-001/DR-002 後）。

## 8. 整合決策（初始）
決策：擴充既有 Insights Portal（mono repo 模組）而非獨立新 UI。理由：
1. 重用元件與指標呈現降低重複。  
2. 既有 persona 與 threshold 機制加速交付。  
3. 單一部署成品降低維運成本。  
4. 未來衍生 artifacts 可復用 normalization 流程。  
風險：程式碼庫複雜度上升；緩解：依領域路由與延遲載入。  

## 9. 追溯
對應平台需求：UI-FR-033/034/035 ↔ FR-041/042；UI-FR-060 ↔ 追溯性目標 ≥95%；UI-FR-062/063 支援 CLI 過渡。

---
文件結束。
