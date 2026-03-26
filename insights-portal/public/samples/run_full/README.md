# Samples — run_full

Use this folder to test multi-run compare, CSV parsing, and error handling.

Files:
- `outputs/ragas_enhanced_evaluation_results_20250902.json` — portal-ready JSON with items.
- `outputs/testset_with_rag_responses_20250902.csv` — CSV for the Worker CSV pipeline.
- `outputs/broken_malformed.json` — syntactically valid JSON but semantically invalid (e.g., metric type wrong) to trigger schema validation errors (filename + row/offset handling).
- `outputs/detail_link_sample.json` — includes external URLs in contexts for safe-preview testing.
