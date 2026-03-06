# RAG 評估平台微服務化需求說明書

版本：0.1（草稿）  
狀態：徵求審閱  
日期：2025-09-09  
負責：平台工程組  

---
## 1. 願景
將現有單一 CLI (`run_pipeline.py`) 之領域導向 RAG 評估流程，轉型為模組化、API 驅動、容器化的微服務平台；各階段（文件匯入 → 前處理 → 測試集生成 → RAG 評估 → 洞察分析 → 報告輸出）分離部署、標準化契約、可觀測、可插拔，並與外部知識管理（KM）REST API 及既有 Insights Portal 前端無縫整合。

## 2. 目標（SMART）
1. 於三個迭代內以 ≥6 個邊界情境服務取代單一 CLI。  
2. 端到端工作流程以非同步工作（submit → status → result）API 運行，控制面平均延遲 < 2 秒。  
3. Stateless 服務可水平擴充到 ≥3 副本而不需程式修改。  
4. 追溯性：≥95% 測試樣本結果可回溯原始文件切片與評分規則。  
5. 提供英文與繁體中文雙語需求與 API 文件。  

## 3. 利害關係人
- 平台工程師：建置與維運服務。  
- 資料/ML 工程師：整合自訂 LLM、Embedding、評估指標。  
- QA / 評估分析師：啟動評估並檢視結果。  
- 產品經理：透過 Portal 監控 KPI 趨勢。  
- 資安/法遵：稽核存取控制與資料保護。  
- 外部整合者：從 KM 推送文件或訂閱事件。  

## 4. 詞彙表
- KM API：外部知識管理 REST 端點，提供文件中繼資料與內容。  
- 測試集（Testset）：結構化 Q/A + 上下文 + 中繼資料集合。  
- 評估執行（Evaluation Run）：產出每樣本指標與彙總 KPI 的不可變記錄。  
- Persona：代表使用者角色之設定，影響情境生成。  
- Scenario：由 persona + 查詢意圖 + 約束組合之評估情境。  
- Insights Portal：既有 React/Vite 分析前端。  
- 工作編排（Orchestration）：多階段非同步狀態管理。  

## 5. 服務拆分（初稿）
| 服務           | 職責                                | 主要端點                                 | 資料儲存      | 擴充    | 備註         |
|----------------|-------------------------------------|------------------------------------------|---------------|---------|--------------|
| gateway-api    | 統一入口、驗證、路由、OpenAPI 聚合     | /v1/*                                    | n/a           | 無狀態  | 可延後       |
| ingestion-svc  | 接收文件引用、呼叫 KM、保存原始與雜湊 | POST /documents GET /documents/:id       | 物件儲存 + DB | CPU/I/O | 去重 + 版控  |
| processing-svc | 切片、正規化、語言偵測、Embedding      | POST /process-jobs GET /process-jobs/:id | 向量庫 + DB   | CPU/GPU | 冪等切片流程 |
| testset-gen-svc | 問題/情境/Persona 生成（configurable|ragas|hybrid） | POST /testset-jobs GET /testset-jobs/:id | DB + 物件儲存 | GPU/LLM | 策略外掛化 |
| kg-builder-svc | 知識圖譜節點/關係建構 | POST /kg-jobs GET /kg/:id | 圖資料庫/文件庫 | CPU/Embedding | 可與 processing 合併 |
| eval-runner-svc | 呼叫目標 RAG、計算指標與旗標 | POST /eval-runs GET /eval-runs/:id WS /eval-runs/:id/stream | DB + metrics | CPU/網路 | 指標登錄機制 |
| insights-adapter-svc | 轉換輸出成 Portal 相容構件 | POST /exports GET /exports/:id | 物件儲存 | 無狀態 | Schema 對齊 |
| reporting-svc | 產製 HTML/PDF 高層與技術報告 | POST /reports GET /reports/:id | 物件儲存 | CPU | Headless 瀏覽器 |
| orchestrator-svc | DAG / 狀態機、重試、SLA | POST /workflows GET /workflows/:id | Workflow DB | 無狀態 | 可採外部引擎 |
| authz-svc(未來) | RBAC、Token、範圍密鑰 | POST /tokens | DB | 無狀態 | 第二階段 |

## 6. 高階流程
1. 匯入：POST /documents（km_id, version）。  
2. 前處理：產出切片+Embedding → 事件發佈。  
3. （選）KG 建構。  
4. 測試集生成。  
5. 評估執行：呼叫 RAG 系統、計算指標。  
6. 洞察轉接：輸出 Portal 相容 artifacts。  
7. 報告生成：HTML/PDF。  
8. 編排：追蹤狀態與重試。  

## 7. 外部整合
### 7.1 KM REST API（佔位）
- Base: https://km.example.com/api  
- GET /documents/:id  
- GET /documents/:id/content  
- GET /documents/:id/metadata  
- 錯誤模型: { error_code, message, retryable }  

### 7.2 目標 RAG 系統
POST /query { query, top_k, persona, scenario_id, trace } → { answer, contexts[], latency_ms, model, tokens }  

### 7.3 Insights Portal
- 消費 artifacts：evaluation_items.json、kpis.json、thresholds.json、personas.json、run_meta.json  
- 對齊 `schemas.ts` 正規化規則。  

### 7.4 衍生資源 API（佔位 / 延後）
目的（延後 – DR-001 / DR-002）：為評估執行產生之衍生產物（如：`chunk_index`, `kg_summary`, `evaluation_run_metrics`, `testset_schema`）提供統一存取介面。啟用延至治理與留存策略確立後。

提議端點（全部延後）：
| 方法   | 路徑                      | 說明                                   | 備註                                         |
|--------|---------------------------|----------------------------------------|----------------------------------------------|
| POST   | /derivatives              | 建立衍生資源（中繼資料 +（可選）內容 URI） | 以 (resource_type, source_run_id, hash) 冪等 |
| GET    | /derivatives/:id          | 取得衍生資源中繼資料                   | 404 若不存在                                 |
| GET    | /runs/:run_id/derivatives | 列出特定 run 衍生資源                  | 可依 resource_type 過濾                      |
| DELETE | /derivatives/:id          | （延後）軟刪除（墓碑）                     | 需 admin 角色                                |

OpenAPI YAML 佔位（Phase ≥3）：
```
paths:
	/derivatives:
		post:
			summary: Create derivative (Deferred)
			operationId: createDerivative
			tags: [derivatives]
			requestBody:
				required: true
				content:
					application/json:
						schema:
							$ref: '#/components/schemas/DerivativeCreateRequest'
			responses:
				'202': { description: Accepted }
	/derivatives/{id}:
		get:
			summary: Get derivative (Deferred)
			operationId: getDerivative
			tags: [derivatives]
			parameters:
				- name: id
					in: path
					required: true
					schema: { type: string, format: uuid }
			responses:
				'200': { description: OK }
				'404': { description: Not Found }
	/runs/{run_id}/derivatives:
		get:
			summary: List derivatives by run (Deferred)
			operationId: listRunDerivatives
			tags: [derivatives]
			parameters:
				- name: run_id
					in: path
					required: true
					schema: { type: string, format: uuid }
				- name: resource_type
					in: query
					required: false
					schema: { type: string }
			responses:
				'200': { description: OK }
```

備註：
- 安全性：需未來 authz-svc（Token + RBAC）。
- 儲存：物件儲存（內容） + 關聯/KV 中繼索引。
- 留存：受第 14 節開放問題（DR-001）策略影響。
- DR-002 部分解決將於第 8.8 節提供草稿 Schema；列舉值未定稿。


## 8. 資料契約
(與英文版對應，不重複列出全部 JSON 範例)  
- Document / Chunk / Test Sample / Evaluation Result / KPI Aggregation 結構相同。  

### 8.6 執行中繼資料 Run Meta（擴充）
```
{
	"run_id":"uuid",
	"created_at":"ts",
	"pipeline_version":"string",
	"metrics_version":"v1",
	"report_html_url":"https://object-store/runs/<id>/report.html",
	"report_pdf_url":"https://object-store/runs/<id>/report.pdf",  // 可選
	"export_profile":"default",
	"extras": {}
}
```

### 8.7 匯出摘要 Export Summary（選用）
```
{
	"run_id":"uuid",
	"kg_summary":{"node_count":120,"relationship_count":340},      // 有 KG 時存在
	"persona_stats":{"persona_count":5,"scenario_count":12},       // 有 persona 時存在
	"feature_flags":{"kg_summary_export":true,"persona_stats":true}
}
```

### 8.8 衍生資源（延後草稿）
狀態：草稿（DR-002 部分解決；最終契約延至設計階段）。此 Schema 捕捉未來由評估執行衍生之產物的最小共同中繼欄位。列舉值與留存政策尚未定稿。

範例：
```
{
	"derivative_id": "uuid",
	"source_run_id": "uuid",
	"resource_type": "kg_summary",            // 候選 enum，非最終
	"version": "v0",
	"created_at": "ts",
	"hash": "sha256",
	"pii_classification": "none|low|moderate|high",
	"content_uri": "s3://bucket/runs/<run>/derivatives/kg_summary.json",
	"size_bytes": 2048,
	"status": "available|expired|tombstoned",
	"metadata": {
		"node_count": 120,
		"relationship_count": 340
	}
}
```

JSON Schema（草稿）：
```
{
	"$schema": "https://json-schema.org/draft/2020-12/schema",
	"$id": "https://example.com/schemas/derivative-resource.schema.json",
	"title": "DerivativeResource",
	"type": "object",
	"required": [
		"derivative_id",
		"source_run_id",
		"resource_type",
		"created_at",
		"hash",
		"content_uri",
		"status"
	],
	"properties": {
		"derivative_id": { "type": "string", "format": "uuid" },
		"source_run_id": { "type": "string", "format": "uuid" },
		"resource_type": { "type": "string", "description": "暫定列舉（例：chunk_index, kg_summary, evaluation_run_metrics, testset_schema）" },
		"version": { "type": "string", "default": "v0" },
		"created_at": { "type": "string", "format": "date-time" },
		"hash": { "type": "string", "description": "完整性 SHA-256（或未來演進）" },
		"pii_classification": { "type": "string", "description": "風險層級；分類法待定" },
		"content_uri": { "type": "string", "format": "uri" },
		"size_bytes": { "type": "integer", "minimum": 0 },
		"status": { "type": "string", "description": "生命週期狀態 (available|expired|tombstoned) - 未封閉集合" },
		"metadata": { "type": "object", "additionalProperties": true }
	},
	"additionalProperties": false
}
```

非目標（此草稿未涵蓋）：
- 最終 resource_type enum。
- 留存與清除策略（依 DR-001 結論）。
- 衍生資源間相互引用。

預計驗證規則（設計階段）：
- （規則）resource_type + source_run_id + hash 必須唯一。
- （規則）status=tombstoned 時 content_uri 不可公開存取。
- （規則）pii_classification != 'none' 時需提升存取權限。


## 9. 功能性需求（EARS）
(僅列出需求 ID 與核心文字；細節同英文 FR-001 ~ FR-036)
- FR-001 ~ FR-005：文件匯入與去重、重試、校驗。  
- FR-006 ~ FR-009：處理工作、切片、Embedding、錯誤標記。  
- FR-010 ~ FR-012：KG 建構與關係策略。  
- FR-013 ~ FR-016：測試集生成模式、Persona、上限裁剪、問題唯一。  
- FR-017 ~ FR-022：評估執行、指標計算、人工審查旗標、不變性。  
- FR-023 ~ FR-025：Portal 正規化與匯出。  
- FR-026 ~ FR-027：報告產出（HTML/PDF）。  
- FR-028 ~ FR-030：工作編排、重試與完成事件。  
- FR-031 ~ FR-036：健康檢查、結構化日誌、進度、OpenAPI、避免阻塞、熱重載。  

### 9.10 報告與 Portal 整合（新增）
FR-037 (EVENT) 當報告生成完成時，reporting-svc 應寫入 report_html_url 與（若存在）report_pdf_url 至 run_meta.json。  
	驗收：run_meta.json URL 可 200 回應。  
FR-038 (UBI / 可選) 啟用 kg_summary_export 時，insights-adapter-svc 應輸出 export_summary.json 含 kg_summary。  
	驗收：node_count、relationship_count >0 且與 KG 資料一致。  
FR-039 (UBI / 可選) 啟用 persona_stats_export 時，insights-adapter-svc 應輸出 persona_stats。  
	驗收：persona_count 與 personas.json 筆數一致；scenario_count 與 scenarios.json 筆數一致。  
FR-040 (UBI) 平台應提供報告下載引用，Portal 可由 run_meta.report_pdf_url（或 HTML 後備）顯示下載按鈕。  
	驗收：存在至少一個 report_*_url；點擊取得 >5KB 檔案。  

### 9.11 KM 儲存（初始範圍）（新增）
FR-041 (EVENT / Phase 1.5) 當 testset 生成完成 (status=completed) 時，testset-gen-svc 應輸出一個 KM 匯出摘要（testset_schema v0）至 KM 佇列/佔位，內容含：testset_id, sample_count, persona_count, scenario_count, generation_method, schema_version。  
	驗收：KM 暫存訊息可見；payload 通過草稿 Schema；不含完整題目文字。  
FR-042 (EVENT / Phase 1.5) 當 KG 建構完成時，kg-builder-svc 應輸出 KM 匯出摘要：kg_id, node_count, relationship_count, source_document_ids[], build_profile_hash。  
	驗收：摘要送達 KM 暫存；數量與內部 graph.json 一致。  
（延後）其他衍生物（evaluation_run_metrics, chunk_index 等）待 DR-001/DR-002 結論。  

### 9.12 UI 整合（新增）
範圍：
統一操作型 UI（參見 `requirements.ui.md` 與 `requirements.ui.zh.md`）採擴充既有 Insights Portal 方式，而非獨立重建。提供依角色之跨生命週期監控與控制（Documents → Processing → KG → Testsets → Evaluations → Insights → Reports），並揭露 KM 匯出摘要（FR-041/042）及未來衍生資源佔位。

對應映射（節選）：
- UI-FR-009..012 對應匯入/前處理可視性（FR-001..009）。
- UI-FR-016..018 呈現 KG 指標與關係（FR-010..012）。
- UI-FR-019..022 驅動測試集生成參數（FR-013..016）。
- UI-FR-023..026 監控評估與指標（FR-017..022, 37..40）。
- UI-FR-030..032 映射報告工件（FR-026..027, 37..40）。
- UI-FR-033..035 顯示 KM 摘要（FR-041..042）。
- UI-FR-060 支援追溯性目標（目標 #4 ≥95%）。
- UI-FR-062..064 協助 CLI 過渡（第 13 節）。
- UI-FR-065..067 保留衍生資源空間以待 DR-001 / DR-002。

非功能對齊：
- UI-NFR-001 補強平台可用性目標。
- UI-NFR-005 強化 trace_id 傳遞可見性（第 12 節）。
- UI-NFR-006 保持前端 bundle 體積以符合理想效能。

開放問題（UI 範圍）集中於 UI 需求文件；任何結論若影響後端契約需同步調整本節描述。

## 10. 非功能需求
| 類別   | 需求                          | 驗收          |
|--------|-------------------------------|---------------|
| 效能   | eval-runner 每秒 ≥10 簡單樣本 | 基準測試報告  |
| 擴充   | 水平擴充吞吐線性 ±25%         | 壓力測試結果  |
| 可靠   | 編排重啟恢復 ≤1 步重複        | Chaos 測試    |
| 可用   | 核心服務月可用 ≥99%           | 監控儀表板    |
| 安全   | 啟用 auth 時拒絕未授權        | 測試用例      |
| 可觀測 | /metrics 提供 Prometheus 指標 | Scrape 驗證   |
| 追溯   | 全流程 trace_id 傳遞          | 日誌抽樣      |
| 維護   | 關鍵模組測試覆蓋 ≥80%         | CI 報告       |
| 國際化 | 匯出支援 en, zh-TW            | Artifact 比對 |
| 擴充性 | 測試集策略外掛熱插拔          | 動態載入測試  |
| 留存   | 過期文件每日清理              | 模擬排程      |
| 合規   | PII 遮罩                      | 日誌檢視      |

## 11. 安全與隱私
- Token 驗證、可選 mTLS。  
- 秘密管理：不寫入日誌。  
- PII 偵測與遮罩。  
- RBAC（第二階段）。  

## 12. 可觀測性
- 結構化 JSON 日誌。  
- Prometheus 指標 + OpenTelemetry Trace。  
- Grafana 儀表板。  

## 13. 遷移策略
0：凍結 CLI 功能。  
1：抽離 ingestion / processing / evaluation。  
2：加入 KG & testset-gen + Portal 匯出。  
3：報告 & gateway & RBAC。  
4：效能優化與快取。  

### 13.1 里程碑（指標性）
| Phase     | 主題       | 主要交付物                                                                      | 退出條件                              |
|-----------|------------|---------------------------------------------------------------------------------|---------------------------------------|
| 0         | 穩定化     | CLI 強化、基礎日誌                                                               | Legacy 執行無 P1 缺陷                 |
| 1         | 核心抽離   | ingestion-svc, processing-svc, eval-runner-svc (FR 1-9,17-22)、簡易 orchestrator | 透過 API 產出 evaluation_items.json   |
| 2         | 能力擴充   | testset-gen-svc, kg-builder-svc, insights-adapter (FR 10-16,23-25,37-39 部分)   | Portal 可直接消費標準化 artifacts     |
| 3         | 呈現與安全 | reporting-svc, gateway-api, 初始 authz-svc, run_meta 報告 URL (FR 26-27,37,40)  | HTML/PDF 可下載且受控存取             |
| 3.5（延後） | 衍生基礎   | Derivatives OpenAPI 佔位（唯讀列表）、/derivatives create 內部原型                 | kg_summary 成功以 derivative 形式儲存 |
| 4         | 效能最佳化 | 快取、批次、向量索引優化                                                          | p95 延遲與吞吐達標                    |
| 5（延後）   | 衍生 GA    | /derivatives 完整（除硬刪除）、留存策略、權限分層                                   | DR-001/DR-002 關閉，策略文件核准       |


## 14. 開放問題
 - 工作流引擎選擇：自製 vs Temporal vs Prefect（評估中）。  
 - 向量儲存：本地 FAISS vs 受管（pgvector, Milvus）尚未定案。  
 - 圖資料庫：是否需要外部圖（Neo4j）或嵌入式結構足夠？  
 - 外部 RAG 系統速率限制策略？  
 - 人類回饋（Human Feedback）儲存模型尚未定稿。  
	- （延後）評估衍生資源回推 KM (POST /documents/:id/derivatives) 範圍與留存（DR-001）。  
	- （延後）衍生資源型別集合（chunk_index, kg_summary, evaluation_run_metrics, testset_schema）共同中繼欄位最終列舉與留存（DR-002）。  
		*更新：* 第 8.8 節提供部分草稿；列舉與留存仍延後。
	- （追蹤）KM DB 整合初始僅限 testset summary 與 KG summary 匯出（FR-041, FR-042）；更廣推送延後。
	- （已解決）KG 視覺化套件選擇 → 採 Cytoscape.js（延遲載入），詳見設計文件第 20 節；影響 GET /ui/kg/{kg_id}/summary 回應欄位定義。
- Workflow 引擎選擇？  
- 向量庫選項？  
- 圖資料需求層級？  
- RAG 系統頻寬 / Rate limit？  
- 人工標註資料模型？  
 - （延後）是否上傳評估衍生物回 KM（POST /documents/:id/derivatives）之範圍與留存策略 (DR-001)。  
 - （延後）潛在衍生資源類型（chunk_index、kg_summary、evaluation_run_metrics、testset_schema）共用核心欄位 {derivative_id, source_run_id, resource_type, created_at, hash, pii_classification}，最終契約延至設計階段 (DR-002)。  
	*更新：第 8.8 節提供部分草稿；列舉與留存待後續確定。*
 - KM 初始範圍僅匯出 testset 與 KG 摘要（FR-041, FR-042）；其餘衍生上傳待 DR-001/DR-002 結論。  

## 15. 假設
- KM API ID 穩定。  
- 內網可存取 RAG。  
- 部署於 Kubernetes / 開發使用 Docker Compose。  

## 16. 風險與緩解
| 風險         | 影響         | 緩解                |
|--------------|--------------|---------------------|
| LLM 延遲波動 | 吞吐下降     | 併發控制 + 快取     |
| 指標漂移     | 歷史不可比   | 指標版本 + 模型雜湊 |
| 過度微服務   | 維運成本     | 週期性整併檢視      |
| 匯入重複     | 儲存膨脹     | 雜湊去重            |
| 外掛安全     | 任意程式風險 | 沙箱 / 簽章         |

## 17. 建議（超出原始想法）
- 事件匯流排（Kafka/NATS）解耦。  
- 統一 Job Status Schema。  
- Feature Flag (OpenFeature)。  
- Python SDK + 向後相容 CLI。  
- RAG 回應結果快取。  
- 場景重播（Replay）能力。  
- 可重現性：測試集種子管理。  

## 18. 追溯矩陣（樣板）
| 需求 ID    | 類型   | 對應目標(編號) | 來源章節            | 關聯 UI（若後端）     | 驗證/測試參考                  |
|------------|--------|----------------|---------------------|---------------------|--------------------------------|
| FR-001     | 後端   | 目標2, 目標3   | 9.1 匯入            | UI-FR-009,010       | T-Ingest-Create                |
| FR-002     | 後端   | 目標2          | 9.1 匯入            | UI-FR-009           | T-Ingest-Checksum              |
| FR-010     | 後端   | 目標2          | 9.3 KG              | UI-FR-016..018      | T-KG-Build-MinRelationships    |
| FR-013     | 後端   | 目標2, 目標4   | 9.4 測試集生成      | UI-FR-019..022      | T-Testset-Generation-Config    |
| FR-017     | 後端   | 目標2, 目標4   | 9.5 評估執行        | UI-FR-023..026      | T-Eval-Run-Lifecycle           |
| FR-026     | 後端   | 目標2          | 9.7 報告            | UI-FR-030..032      | T-Report-HTML-Generation       |
| FR-037     | 後端   | 目標2          | 9.10 報告整合       | UI-FR-030..032      | run_meta.json URL 檢查         |
| FR-041     | 後端   | 目標4          | 9.11 KM 儲存        | UI-FR-033..035      | KM testset_summary 匯出驗證    |
| FR-042     | 後端   | 目標4          | 9.11 KM 儲存        | UI-FR-033..035      | KM kg_summary 匯出驗證         |
| UI-FR-009  | 前端   | 目標2          | UI 5.3 文件匯入視圖 | FR-001..005         | UI 測試：文件列表顯示 checksum  |
| UI-FR-016  | 前端   | 目標2, 目標4   | UI 5.5 KG 儀表      | FR-010..012         | UI 測試：KG 指標面板            |
| UI-FR-019  | 前端   | 目標2, 目標4   | UI 5.6 測試集控制台 | FR-013..016         | UI 測試：Testset 表單 hash 顯示 |
| UI-FR-023  | 前端   | 目標2, 目標4   | UI 5.7 評估協調     | FR-017..022         | WebSocket 串流整合測試         |
| UI-FR-030  | 前端   | 目標2          | UI 5.9 報告與工件   | FR-026..027,037,040 | UI 測試：報告預覽               |
| UI-FR-033  | 前端   | 目標4          | UI 5.10 KM 摘要     | FR-041,042          | UI 測試：KM 摘要差異高亮        |
| UI-FR-060  | 前端   | 目標4          | UI 5.18 追溯        | FR-017..022         | 血緣鏈顯示測試                 |
| UI-FR-062  | 前端   | 目標2(遷移)    | UI 5.19 CLI 過渡    | 多個                | CLI 指令提示測試               |
| UI-FR-065  | 前端   | 目標4(未來)    | UI 5.20 衍生預留    | DR-001/002 延後     | 衍生分頁佔位顯示               |
| UI-NFR-001 | UI-NFR | 目標3          | UI NFR              | n/a                 | 可用性合成檢測                 |
| UI-NFR-005 | UI-NFR | 目標4          | UI NFR              | FR-031..033         | 90% API 呼叫含 trace_id 抽樣   |
| UI-NFR-006 | UI-NFR | 目標2          | UI NFR              | n/a                 | Bundle 大小 CI 門檻            |
| ...        | ...    | ...            | ...                 | ...                 | ...                            |

## 19. 後續步驟
1. 審閱並確認範圍。  
2. 決定 Phase 1 需求（FR-001~009, 017~022, 031~035 + 核心 NFR）。  
3. 產出 OpenAPI 骨架。  
4. 定義事件 Schema 與共享程式庫。  
5. 建立服務 CI 模板。  

---
文件結束。
