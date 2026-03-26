import importlib
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

SERVICE_MODULES = [
    ("services.ingestion.main", "ingestion-service"),
    ("services.processing.main", "processing-service"),
    ("services.testset.main", "testset-service"),
    ("services.eval.main", "eval-service"),
    ("services.reporting.main", "reporting-service"),
    ("services.adapter.main", "adapter-service"),
]


def _load_module(path: str) -> ModuleType:
    module = importlib.import_module(path)
    return importlib.reload(module)


@pytest.mark.parametrize("module_path,service_name", SERVICE_MODULES)
def test_health_endpoint_exposes_service_name(module_path: str, service_name: str) -> None:
    module = _load_module(module_path)
    app = getattr(module, "app")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "service": service_name}
    trace_id = response.headers.get("x-trace-id")
    assert trace_id is not None
    assert len(trace_id) >= 8


def test_ingestion_validation_error_envelope() -> None:
    module = _load_module("services.ingestion.main")
    app = getattr(module, "app")

    with TestClient(app) as client:
        response = client.post("/documents", json={})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "request_validation_failed"
    assert body["trace_id"]
    assert isinstance(body["details"], list)
