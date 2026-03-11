# 待改進與優化事項
1. Observability (Langfuse/LangSmith) -> 可透過環境變數啟用 LangSmith Tracing。
2. Async Batch -> Ragas 本身已有一定程度的非同步支持，我們需要確保在使用 `run_async`。
3. 統一測試集分布 -> 設定 `TestsetGenerator` 的 `distributions`。
4. 抽離 Prompt -> 新增 `prompts/` 目錄。
