# Portal-ready Summary Generation and UI Test

This guide shows how to generate a portal-ready summary JSON from evaluation artifacts and test it in the Insights Portal UI.

## Prerequisites
- Python 3.8+
- Node.js LTS (for the portal UI)
- A run folder with `evaluations-pre/` (e.g., `eval-pipeline/outputs/<run-id>/evaluations-pre`)

## Generate a portal-ready summary JSON
- The converter aggregates per-question metrics from `ragas_enhanced_detailed_calculations_*.json` and produces `{ items: [...] }`.

```bash
python3 /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/convert_to_portal_summary.py \
  /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/outputs/<run-id>/evaluations-pre
# Output: /mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/outputs/<run-id>/portal/ragas_enhanced_evaluation_results_<timestamp>_portal.json
```

Notes
- Metrics supported: ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerSimilarity, ContextualKeywordMean
- Snake_case variants are auto-mapped. Missing `id` falls back to common fields; per-item latency is optional.

## Test in the Insights Portal
1) Start the portal
```bash
cd /mnt/d/workspace/domain-specific-llm-eval/insights-portal
npm install
npm run dev
# Open http://localhost:5173
```

2) Optional: thresholds profile
```bash
mkdir -p public/profiles
cp -r ../profiles/* public/profiles/
```

3) Load the generated summary
- Click "選擇 JSON 檔載入 run" and choose the JSON in `outputs/<run-id>/portal/`.
- Or click "選擇資料夾並掃描 runs" and select `outputs/<run-id>/`, then click 載入.

Expected
- Verdict banner shows Ready/At Risk/Blocked; Items > 0
- KPI cards render metric values; latency p50/p90 shows N/A if latency is not provided
- Sorting by threshold gap works; editing thresholds updates verdict/gaps instantly

## Troubleshooting
- 0 items: The source file may be a report-type JSON without per-item metrics. Use the converter against `ragas_enhanced_detailed_calculations_*.json` or adjust the pipeline to emit the summary.
- Missing metrics: Only supported keys are aggregated; add mapping if your pipeline uses different names.
- Latency: Add a `latencyMs` (or alias) field per item if you want latency stats in the UI.

## CI suggestion (optional)
- Add a validation step to assert the emitted summary JSON parses and yields non-zero items for supported metrics.
