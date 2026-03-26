import json
import logging
import sys
import time
import uuid
from typing import Any, Dict
from .config import settings

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": settings.service_name,
        }
        if hasattr(record, "trace_id"):
            base["trace_id"] = getattr(record, "trace_id")

        context: Any = getattr(record, "context", None)
        if isinstance(context, dict):
            base.update(context)

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=False)

def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root.handlers = [handler]

configure_logging()

def get_trace_id() -> str:
    return uuid.uuid4().hex
