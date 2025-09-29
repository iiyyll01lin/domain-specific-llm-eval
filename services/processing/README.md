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
