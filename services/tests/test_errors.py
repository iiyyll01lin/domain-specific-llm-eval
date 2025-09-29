from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel

from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware


class PayloadModel(BaseModel):
    name: str


app = FastAPI()
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.get("/service-error")
async def trigger_service_error():
    raise ServiceError("demo_error", "Demo failure", http_status=status.HTTP_418_IM_A_TEAPOT)


@app.post("/validate")
async def validate_payload(payload: PayloadModel):
    return payload


@app.get("/boom")
async def trigger_generic_error():
    raise RuntimeError("unexpected failure")


client = TestClient(app, raise_server_exceptions=False)


def test_service_error_envelope_contains_trace_id():
    response = client.get("/service-error")
    assert response.status_code == status.HTTP_418_IM_A_TEAPOT
    trace_id = response.headers.get("x-trace-id")
    assert trace_id
    body = response.json()
    assert body == {
        "error_code": "demo_error",
        "message": "Demo failure",
        "trace_id": trace_id,
    }


def test_validation_error_handler_returns_structured_payload():
    response = client.post("/validate", json={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    trace_id = response.headers.get("x-trace-id")
    assert trace_id
    body = response.json()
    assert body["error_code"] == "request_validation_failed"
    assert body["message"] == "Request payload validation failed"
    assert body["trace_id"] == trace_id
    assert isinstance(body.get("details"), list)
    assert body["details"], "validation details expected"


def test_generic_error_handler_masks_internal_details():
    response = client.get("/boom")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    trace_id = response.headers.get("x-trace-id")
    assert trace_id
    body = response.json()
    assert body == {
        "error_code": "internal_error",
        "message": "Internal server error",
        "trace_id": trace_id,
    }
