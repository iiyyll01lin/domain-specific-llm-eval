# 任務治理驗證工具 (tasks governance validator)

此文件說明 `scripts/validate_tasks.py` 使用方式、參數、輸出與 CI 整合策略。提供漸進與嚴格兩種採用模式。

---
## 1. 目的
確保 `tasks.md` 與 `tasks.zh.md` 之間：
- 子任務 (a–d) 列表雙語一致 (Parity)
- 每個子任務具備治理區塊 (Governance Block)
- 治理區塊包含必填欄位：`status, owner, priority, estimate, risk, mitigation, adr_impact, ci_gate, dod`
- 產生矩陣 JSON 供前端或治理面板使用

---
## 2. 參數
| 參數                          | 說明                                                   | 範例                                                           |
|-------------------------------|--------------------------------------------------------|----------------------------------------------------------------|
| `--focus`                     | 逗號分隔父任務 ID (僅檢查此集合；格式 TASK-\d{3})       | `--focus TASK-015,TASK-020`                                    |
| `--strict`                    | 嚴格模式：即使 focus 集合通過，只要全域存在缺口仍失敗    | `--focus TASK-015 --strict`                                    |
| `--output-dir`                | JSON 輸出目錄（預設 scripts/）                           | `--output-dir build/validation`                                |
| `--report-md <檔名>`          | 產出治理 Markdown 報表 (排序/狀態徽章/Parity 標記)     | `--report-md governance_report.md`                             |
| `--report-md-include-missing` | 報表尾端列出缺少治理 block 或欄位 (需搭配 --report-md) | `--report-md governance_report.md --report-md-include-missing` |
| `--include-all-tasks`         | 報表中加入所有主任務列 (即便非 focus；以佔位列呈現)    | `--report-md governance_report.md --include-all-tasks`         |

未提供 `--focus` 時使用內建父任務集合：`015,020,021,030,031,032,033,034,062,126,130,132`。

---
## 3. 輸出檔案
| 檔案                      | 內容                                                                                                                  |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `parity_matrix.json`      | 各父任務 EN/ZH 子任務列表與是否 match                                                                                 |
| `governance_gaps.json`    | EN / ZH 缺少治理欄位之子任務對應欄位列表                                                                              |
| `governance_matrix.json`  | 子任務治理摘要（owner/priority/status & 是否有 block）                                                                  |
| `validation_summary.json` | 驗證摘要：focus_parents、嚴格模式、缺口計數、退出碼前提                                                                   |
| `governance_report.md`    | (選用) 依優先級排序之 Markdown 矩陣 + 變更摘要 + (可選) 缺口列表；含分類：Parent-Skeleton-Missing 與 Missing Governance |

> 若使用 `--include-all-tasks`，`governance_matrix.json` 會額外包含每個主任務一列佔位資料，`subtask` 與 `parent` 欄位相同，並新增布林欄位 `is_parent_only:true`。

---
## 4. 退出碼 (Exit Codes)
| Code | 意義                       |
|------|----------------------------|
| 0    | 全部通過                   |
| 1    | 治理缺口 (governance gaps) |
| 2    | Parity 不一致              |
| 3    | 同時存在 parity 與治理缺口 |
| 4    | 參數錯誤 (如非法 TASK ID)  |

---
## 5. 使用範例
### 全量檢查
```
python scripts/validate_tasks.py
```
### 聚焦檢查 (漸進採用)
```
python scripts/validate_tasks.py --focus TASK-015,TASK-020
```
### 嚴格模式 (CI Gate 轉換階段)
```
python scripts/validate_tasks.py --strict
```
### 聚焦 + 嚴格 (聚焦報告 + 全域品質門檻)
```
python scripts/validate_tasks.py --focus TASK-015 --strict
```
### 自訂輸出目錄
```
python scripts/validate_tasks.py --output-dir build/governance
```

### 產出 Markdown 報表
```
python scripts/validate_tasks.py --report-md governance_report.md
```
### 報表並列出缺少治理欄位 / block
```
python scripts/validate_tasks.py --report-md governance_report.md --report-md-include-missing
```
### 產出含所有主任務（含非 focus）之報表
```
python scripts/validate_tasks.py --report-md governance_report.md --include-all-tasks
```
> 併用 `--report-md-include-missing` 時，若出現分類：
> - `Parent-Skeleton-Missing`：表示【父任務本身尚無治理區塊】但其下至少一個子任務已有治理；屬於「應補父層骨架」類。
> - `Missing Governance`：任務（父或獨立）完全沒有任何治理區塊，且也無子任務已覆蓋；屬於較原始缺口。

---
## 6. CI 整合建議 (GitHub Actions 範例)
```yaml
jobs:
  governance-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install minimal deps (如需)
        run: |
          pip install -r requirements.txt || true
      - name: Governance Validation (Focused Progressive)
        run: |
          python eval-pipeline/scripts/validate_tasks.py --focus TASK-015,TASK-020 --report-md governance_report.md
      - name: Upload Governance Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: governance-validation
          path: |
            eval-pipeline/scripts/parity_matrix.json
            eval-pipeline/scripts/governance_gaps.json
            eval-pipeline/scripts/governance_matrix.json
            eval-pipeline/scripts/validation_summary.json
            eval-pipeline/scripts/governance_report.md
```
轉為嚴格階段時改為：
```
python eval-pipeline/scripts/validate_tasks.py --strict --report-md governance_report.md --report-md-include-missing
```
> 建議：將 `governance_report.md` 作為 PR Artifact，供 Reviewer 快速檢視新增或優先級變更子任務。

---
## 7. 漸進採用策略
1. Phase A：`--focus` 於關鍵路徑 (Processing/Testset/Eval) → 快速修補
2. Phase B：擴展 focus 列表 + 補齊其餘骨架 (Security/Parity/Extensions)
3. Phase C：啟用 `--strict` → 確保未來新增子任務立即覆蓋治理欄位
4. Phase D：前端儀表使用 `governance_matrix.json` 顯示 owner/priority/狀態熱度圖
5. Phase E：使用 `governance_report.md` 快速檢視優先級與變更摘要（PR Review 支援）

---
## 8. 欄位說明
| 欄位       | 目的           | 常見值                         |
|------------|----------------|--------------------------------|
| status     | 任務狀態       | Planned / In-Progress / Done   |
| owner      | 負責團隊       | platform-processing@team 等    |
| priority   | 優先級         | P0 / P1 / P2 / P3              |
| estimate   | 相對點數       | 1p / 2p / 3p                   |
| risk       | 主要風險       | 簡潔句子                       |
| mitigation | 緩解策略       | 測試/架構/防護敘述             |
| adr_impact | 受影響 ADR     | ["ADR-001"]                    |
| ci_gate    | CI 需觸發 Gate | ["unit-tests","perf-baseline"] |
| dod        | 完成定義       | 列出可驗證條目                 |

---
## 9. 常見問題 (FAQ)
Q: 若中英文某子任務未同步？
A: parity_matrix.json 中 `match:false`，退出碼至少為 2 或 3，請於另一語言補齊該子任務行與治理區塊。

Q: 為何 governance_gaps.json 仍列出 EN 缺口？
A: 可能主任務 skeleton 尚未補完；使用 `--focus` 先針對關鍵集合壓縮缺口，再逐步擴展。

Q: 何時啟用嚴格模式？
A: 當 80% 以上父任務已完成所有子任務治理欄位時；否則會造成過多噪音。

Q: 為何報表末尾顯示 `無新增或優先級變更`？
A: 與前一次 `governance_matrix.json` 比較無新子任務或優先級差異。

Q: 報表排序規則？
A: 先依任一語言存在的優先級 (P0→P1→P2→P3→未設定) → parent ID → subtask。

Q: Parity 欄位 `DIFF` 代表什麼？
A: EN/ZH 在 owner 或 priority 至少一項不一致；狀態允許暫時差異。

Q: 狀態徽章含義？
A: `🟡 Planned` 規劃中、`🟠 In-Progress` 進行、`⛔ Blocked` 阻塞、`✅ Done` 完成、`🟢 Verified` 驗證完成。

Q: `Parent-Skeleton-Missing` 與 `Missing Governance` 差異？
A: 前者代表「子任務已有治理，但父任務尚未建立其摘要治理區塊（骨架）」，後者則是該任務（父或獨立）在任一語言皆完全無治理資料。優先先補 Parent-Skeleton 以提供上層匯總指標，再逐步消化大量 Missing Governance。

Q: 是否會因 Parent-Skeleton-Missing 直接失敗退出碼？
A: 目前不會；退出碼仍僅由 parity 與子任務欄位缺口 (governance gaps) 決定。可於未來在嚴格階段新增旗標（例如 `--treat-parent-missing-as-gap`）。

Q: 需要立即補完所有 Missing Governance 嗎？
A: 建議分批：先確保核心主線 (Processing/Testset/Eval/Observability/Security) 與所有已存在子任務的父層骨架齊備，再設定里程碑逐批處理剩餘散佈的父任務。

---
## 10. 後續延伸 (Future Enhancements)
- 增加 `--format sarif` 以便在 PR 中 Inline 註解缺口
- 與 OPA 整合：輸出 rego data JSON 供 policy 查詢
- 產出 HTML 報表 (governance heatmap)
- 增加 CSV 匯出供資料分析
- 報表加入風險欄位統計 (高/中/低)

---
## 11. 驗證矩陣欄位 (governance_matrix.json)
| 欄位                        | 說明                          |
|-----------------------------|-------------------------------|
| subtask                     | 子任務 ID (含字母)            |
| parent                      | 父任務 ID                     |
| en_owner / zh_owner         | 中英文 owner (應一致)         |
| en_priority / zh_priority   | 優先級 (應一致)               |
| en_status / zh_status       | 狀態 (允許不同步但應盡快對齊) |
| en_has_block / zh_has_block | 是否存在治理區塊              |
| is_parent_only              | 是否為主任務佔位列 (無子任務欄位時提供骨架可視) |

---
### 附錄：Markdown 報表 (governance_report.md) 結構
1. 標頭：Focus 集合 + Strict 模式標記
2. 排序規則：P0→P1→P2→P3→未設定
3. 主表格欄位：Subtask / Parent / EN Owner / ZH Owner / EN Pri / ZH Pri / EN Status / ZH Status / EN Block / ZH Block / Parity
4. Coverage Summary：顯示總主任務數、子任務數、EN/ZH 治理覆蓋率、Owner/Priority Parity 比例
5. (若啟用) Parent 佔位列：當使用 `--include-all-tasks` 時，每個主任務會有一列 `is_parent_only`=true 之骨架行，方便檢視哪些父層尚缺治理
4. Change Summary：列出新增子任務與優先級變更；若無則顯示固定訊息
5. Parent-Skeleton-Missing (可選)：父任務尚無治理骨架但子任務已存在（需 `--report-md-include-missing`）
6. Missing Governance (可選)：完全缺治理（需 `--report-md-include-missing`）
7. 狀態徽章 Legend：見 FAQ

#### 分類策略說明
| 分類                    | 觸發條件                                                                                       | 建議處理優先 | 補齊內容                                                     |
|-------------------------|------------------------------------------------------------------------------------------------|--------------|--------------------------------------------------------------|
| Parent-Skeleton-Missing | 有子任務 (至少一個子任務 en_has_block 或 zh_has_block 為 True)，但父任務本身 EN/ZH 均無治理區塊 | 高           | 建立父層治理骨架：摘要範圍、整體風險、整體 ci_gate、共享完成定義 |
| Missing Governance      | 任務（父或獨立）本身無治理區塊，且無任一子任務具備治理                                            | 中           | 若需細分，先決定是否要新增子任務；否則直接建立父層骨架         |

> 建議：在治理版面 (dashboard) 上可將 Parent-Skeleton-Missing 視為「需補摘要」指標，與全域空白任務分離，以降低噪音。

---
## 12. 驗收清單
- [x] 腳本支援 focus/strict
- [x] 產出 4+1 JSON (新增 governance_matrix)
- [x] CI 範例
- [x] 漸進採用策略文件化
- [x] 分類：Parent-Skeleton-Missing vs Missing Governance 文件化

---
如需擴充請於 ISSUE 標記 governance-tool 與 enhancement 標籤。
