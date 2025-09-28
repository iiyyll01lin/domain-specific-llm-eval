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
