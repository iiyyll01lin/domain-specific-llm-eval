from fastapi.testclient import TestClient

from services.processing.main import app


def test_metrics_endpoint_exposes_prometheus_metrics() -> None:
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "processing_embedding_batch_duration_seconds" in body
    assert "processing_embedding_error_total" in body
    assert 'gpu_enabled{service="processing-service"}' in body
