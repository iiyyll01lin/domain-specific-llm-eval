import json
import logging

from fastapi.testclient import TestClient

from services.common.config import settings
from services.common.logging import JsonFormatter
from services.ingestion.main import app


client = TestClient(app)


def test_trace_middleware_logs_request_context(caplog):
    with caplog.at_level(logging.INFO, logger="services.common.middleware"):
        response = client.get("/health")

    assert response.status_code == 200
    trace_id = response.headers.get("x-trace-id")
    assert trace_id, "trace header must be present"

    matching_records = [record for record in caplog.records if record.getMessage() == "request.completed"]
    assert matching_records, "request.completed log not emitted"

    record = matching_records[-1]
    assert getattr(record, "trace_id", None) == trace_id
    assert isinstance(getattr(record, "context", None), dict)

    formatter = JsonFormatter()
    payload = json.loads(formatter.format(record))
    assert payload["trace_id"] == trace_id
    assert payload["http_method"] == "GET"
    assert payload["http_path"] == "/health"
    assert payload["status_code"] == 200
    assert payload["service"] == settings.service_name
    assert payload["duration_ms"] >= 0
