from __future__ import annotations

import math
from typing import Mapping, MutableMapping


def sanitize_metrics(record: Mapping[str, float | None]) -> MutableMapping[str, float | None]:
    sanitized: MutableMapping[str, float | None] = {}
    for key, value in record.items():
        if value is None:
            sanitized[key] = None
            continue
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            sanitized[key] = None
        else:
            sanitized[key] = numeric
    return sanitized


__all__ = ["sanitize_metrics"]
