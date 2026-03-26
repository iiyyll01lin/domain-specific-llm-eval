"""TASK-091: PII & secret redaction utility.

Provides ``redact(text)`` which replaces common PII / secret patterns with
a ``[REDACTED]`` placeholder, suitable for log output and summaries.

Patterns covered:
- Bearer / API tokens (Authorization header values)
- AWS-style access / secret keys
- Passwords in connection strings (password=xxx)
- E-mail addresses
- IPv4 addresses
- Common credit-card number formats (4 groups of 4 digits)
- Social security numbers (NNN-NN-NNNN)
- Phone numbers (+1 NNN-NNN-NNNN style and local)
- Generic high-entropy hex strings of 32+ chars (e.g. API keys)
"""
from __future__ import annotations

import re
from typing import List, Tuple

# (pattern, replacement) pairs — applied in order
_RULES: List[Tuple[re.Pattern, str]] = [
    # Authorization header value
    (re.compile(r"\b(Bearer|Token|ApiKey)\s+[A-Za-z0-9._\-/+]{8,}", re.IGNORECASE), r"\1 [REDACTED]"),
    # password= in connection strings
    (re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE), r"\1[REDACTED]"),
    # AWS-style keys (AKIA… / long alpha-numeric)
    (re.compile(r"\b(AKIA[A-Z0-9]{16})\b"), "[REDACTED]"),
    # Generic high-entropy hex/base64 secrets (32+ char runs)
    (re.compile(r"\b([A-Fa-f0-9]{32,})\b"), "[REDACTED]"),
    # E-mail addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED]"),
    # IPv4
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED]"),
    # Credit card (four groups of 4 digits, optionally separated by space/dash)
    (re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "[REDACTED]"),
    # SSN NNN-NN-NNNN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED]"),
    # Phone +1 NNN-NNN-NNNN or (NNN) NNN-NNNN
    (re.compile(r"(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED]"),
]


def redact(text: str) -> str:
    """Return a copy of *text* with PII and secrets replaced by ``[REDACTED]``."""
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text


def redact_dict(data: dict) -> dict:
    """Recursively redact string values in *data* (shallow copy)."""
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = redact(v)
        elif isinstance(v, dict):
            result[k] = redact_dict(v)
        else:
            result[k] = v
    return result
