# Priority Next Steps for Domain-Specific LLM Evaluation - Phase 12 (Autonomous Omniverse & Decentralized Edge Intelligence)

隨著 Phase 11 的 AGI Meta-Learning 與 Web3 架構的完備，整個專案已經完全超越傳統管線，成為一個具備自我演化能力的 Agentic OS。
接下來（Phase 12）的發展將專注於 **全宇宙自主算力** 與 **去中心化邊緣智能 (Decentralized Edge Intelligence)**。

## 1. 跨雲自主繁衍架構 (Self-Replicating Cloud Orchestrator) `[Highest Priority]`
- **核心概念**: 目前的基礎設施依賴 Kubernetes / Helm 進行部署。下一步是讓系統能夠自動掃描 AWS/GCP/Azure 的 Spot Instance 價格，並「自主編寫」Terraform / Pulumi 腳本來繁衍自己，直到達到預算上限為止。
- **具體行動**: 實作 `OmniCloudAutoProvisioner`，動態調度多雲資源並自我遷移 Eval-Pipeline。

## 2. DPO / PPO 原生對齊微調 (Native LLM Alignment Pipeline) `[High Priority]`
- **核心概念**: 不只要評估模型，更要在評估出低分的瞬間，自動觸發 Direct Preference Optimization (DPO) 訓練。
- **具體行動**: 在 `RagasEvaluator` 評估完成後，將失敗的 Responses 推入記憶體內的 `Unsloth` / `trl` (Transformer Reinforcement Learning) 佇列進行即時微調。

## 3. 去中心化邊緣評估節點 (Decentralized Edge Mining for RAG) `[Medium Priority]`
- **核心概念**: 取代中央伺服器評估，將 RAG 查詢切割為微小任務，分配給全球的終端設備 (如手機、Edge TPU)。
- **具體行動**: 將評估管線設計為 `WASM (WebAssembly)` 格式，使得瀏覽器或小型設備也能成為分散式的評估驗證節點，實現去中心化的評價資源池。

## 4. 多模態混合現實評估 (Mixed-Reality Multimodal Eval) `[Low Priority]`
- **核心概念**: 突破純文字限制，整合 3D 點雲 (Point Cloud)、空間運算 (Spatial Computing) 以及影像深度，對 LLM 的物理空間認知能力進行立體 RAG 評估。
- **具體行動**: 結合 Unreal Engine 或 Omniverse 引擎，模擬實體環境以生成「空間上下文 (Spatial Context)」進行檢索增強測試。
