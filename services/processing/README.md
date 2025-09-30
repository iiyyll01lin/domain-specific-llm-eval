# Processing Service Stages Overview

## Sentence Tokenization Heuristics
- Supports both English and CJK punctuation terminators (`.?!`, `。！？；`).
- Handles line breaks as hard boundaries and trims surrounding whitespace.
- Preserves leading/trailing closing quotes when they immediately follow sentence punctuation.
- Falls back to a single segment when the upstream mime type is unsupported, logging the downgrade for observability.

## Extraction Stage Highlights
- Downloads objects through the shared object store client with checksum validation.
- Converts PDF payloads via `pdfminer.six`; plain text paths rely on UTF-8 decoding with graceful replacement.
- Performs Unicode normalization (NFKC) and whitespace collapse to provide deterministic downstream inputs.

## Chunk Persistence & Manifest
- Persists chunk text and embedding vectors to `chunks/<document_id>/<profile_hash>/` using the shared object store client.
- Generates a `manifest.json` file capturing per-chunk SHA256 hashes and record counts for integrity validation.
- Uploads validate expected checksums prior to completion and emit `document.processed` events with manifest metadata.

## Embedding Batch Execution
- Embeddings are generated via the shared `EmbeddingBatchExecutor`, which enforces a configurable batch size (default 512).
- Batch execution implements exponential backoff retries for retryable provider errors and opens a circuit breaker after consecutive failures to protect upstream models.
- The effective batch size is additionally bounded by the `PROCESSING_EMBEDDING_MAX_BATCH_SIZE` setting so operators can dial down concurrency without redeploying the service.
- Mismatched vector counts or sequence ordering immediately raise a standardized `ServiceError` to prevent downstream integrity issues.

## Metrics & Error Handling
- Prometheus metrics are exposed at `/metrics`, including `processing_embedding_batch_duration_seconds`, `processing_embedding_batch_size`, and `processing_embedding_error_total` for observability.
- Embedding batch failures emit structured log entries with `code=EMBED_BATCH_FAIL` and increment the error counter.
- All raised `ServiceError` instances return the standardized error envelope `{error_code, message, trace_id}` ensuring UI and orchestrator consumers receive consistent diagnostics.
