#!/usr/bin/env python3
"""
TASK-110: Generate OpenAPI spec stubs for each service.

Produces services/<svc>/openapi.json from route introspection.
Run from repo root: python3 scripts/generate_openapi_specs.py
"""
import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Spec definitions (derived from route inspection of each service main.py)
# ---------------------------------------------------------------------------

SPECS = {
    "processing": {
        "title": "processing-service",
        "version": "0.1.0",
        "description": "Chunking, keyphrase extraction and embedding for ingested documents.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/metrics": {"get": {"summary": "Prometheus metrics", "operationId": "metrics_get",
                "responses": {"200": {"description": "Prometheus text exposition", "content": {"text/plain": {"schema": {"type": "string"}}}}}}},
            "/processing-jobs": {"post": {"summary": "Create processing job", "operationId": "create_processing_job",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ProcessingJobRequest"}}}},
                "responses": {"202": {"description": "Job accepted"}, "422": {"description": "Validation error"}}}},
            "/": {"get": {"summary": "List processing jobs", "operationId": "list_processing_jobs",
                "responses": {"200": {"description": "Job list"}}}},
        },
        "components": {"schemas": {
            "ProcessingJobRequest": {
                "type": "object",
                "required": ["run_id"],
                "properties": {
                    "run_id": {"type": "string"},
                    "chunk_size": {"type": "integer", "default": 512},
                    "chunk_overlap": {"type": "integer", "default": 64},
                }
            }
        }},
    },
    "testset": {
        "title": "testset-service",
        "version": "0.1.0",
        "description": "Testset generation using RAGAS synthesizers.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/metrics": {"get": {"summary": "Prometheus metrics", "operationId": "metrics_get",
                "responses": {"200": {"description": "Prometheus text exposition", "content": {"text/plain": {"schema": {"type": "string"}}}}}}},
            "/testset-jobs": {"post": {"summary": "Create testset generation job", "operationId": "create_testset_job",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TestsetJobRequest"}}}},
                "responses": {"202": {"description": "Job accepted"}, "422": {"description": "Validation error"}}}},
            "/": {"get": {"summary": "List testset jobs", "operationId": "list_testset_jobs",
                "responses": {"200": {"description": "Job list"}}}},
        },
        "components": {"schemas": {
            "TestsetJobRequest": {
                "type": "object",
                "required": ["run_id"],
                "properties": {
                    "run_id": {"type": "string"},
                    "num_questions": {"type": "integer", "default": 10},
                    "persona": {"type": "string", "default": "expert"},
                }
            }
        }},
    },
    "eval": {
        "title": "eval-service",
        "version": "0.1.0",
        "description": "RAG evaluation against testsets using RAGAS metrics.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/metrics": {"get": {"summary": "Prometheus metrics", "operationId": "metrics_get",
                "responses": {"200": {"description": "Prometheus text exposition", "content": {"text/plain": {"schema": {"type": "string"}}}}}}},
            "/eval-jobs": {"post": {"summary": "Create evaluation job", "operationId": "create_eval_job",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/EvalJobRequest"}}}},
                "responses": {"202": {"description": "Job accepted"}, "422": {"description": "Validation error"}}}},
            "/": {"get": {"summary": "List evaluation jobs", "operationId": "list_eval_jobs",
                "responses": {"200": {"description": "Job list"}}}},
        },
        "components": {"schemas": {
            "EvalJobRequest": {
                "type": "object",
                "required": ["run_id", "testset_id"],
                "properties": {
                    "run_id": {"type": "string"},
                    "testset_id": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}, "default": ["faithfulness", "answer_relevancy"]},
                }
            }
        }},
    },
    "reporting": {
        "title": "reporting-service",
        "version": "0.1.0",
        "description": "Report generation (HTML/PDF) from evaluation results.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/": {"get": {"summary": "List report jobs", "operationId": "list_report_jobs",
                "responses": {"200": {"description": "Report list"}}}},
            "/reports": {
                "post": {"summary": "Create report job", "operationId": "create_report_job",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ReportJobRequest"}}}},
                    "responses": {"202": {"description": "Job accepted"}}},
                "get": {"summary": "List completed reports", "operationId": "list_reports",
                    "responses": {"200": {"description": "Report list"}}},
            },
            "/reports/{run_id}/{template}/html": {"get": {"summary": "Get HTML report",
                "operationId": "get_html_report",
                "parameters": [
                    {"name": "run_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "template", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "HTML content"}}}},
            "/reports/{run_id}/{template}/pdf": {"get": {"summary": "Get PDF report",
                "operationId": "get_pdf_report",
                "parameters": [
                    {"name": "run_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "template", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "PDF content"}}}},
        },
        "components": {"schemas": {
            "ReportJobRequest": {
                "type": "object",
                "required": ["run_id"],
                "properties": {
                    "run_id": {"type": "string"},
                    "template": {"type": "string", "default": "default"},
                }
            }
        }},
    },
    "adapter": {
        "title": "adapter-service",
        "version": "0.1.0",
        "description": "Knowledge-map normalization and summary injection adapter.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/": {"get": {"summary": "Service info", "operationId": "root_get",
                "responses": {"200": {"description": "Service metadata"}}}},
            "/normalize": {"post": {"summary": "Normalize KM document",
                "operationId": "normalize_post",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "Normalized document"}}}},
            "/km-summaries/testset": {"post": {"summary": "Inject KM summaries for testset",
                "operationId": "km_summaries_testset",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "Injected testset payload"}}}},
            "/km-summaries/kg": {"post": {"summary": "Inject KM summaries for KG",
                "operationId": "km_summaries_kg",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "Injected KG payload"}}}},
        },
        "components": {"schemas": {}},
    },
    "kg": {
        "title": "kg-service",
        "version": "0.1.0",
        "description": "Knowledge-graph extraction, subgraph sampling, and storage.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/": {"get": {"summary": "List KG jobs", "operationId": "list_kg_jobs",
                "responses": {"200": {"description": "KG job list"}}}},
            "/kg-jobs": {"post": {"summary": "Create KG extraction job",
                "operationId": "create_kg_job",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KgJobRequest"}}}},
                "responses": {"202": {"description": "Job accepted"}}}},
            "/kg-jobs/{kg_id}": {"get": {"summary": "Get KG job status",
                "operationId": "get_kg_job",
                "parameters": [{"name": "kg_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "KG job detail"}, "404": {"description": "Not found"}}}},
            "/kg-jobs/{kg_id}/summary": {"get": {"summary": "Get KG summary statistics",
                "operationId": "get_kg_summary",
                "parameters": [{"name": "kg_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "KG summary"}, "404": {"description": "Not found"}}}},
            "/kg-jobs/{kg_id}/subgraph": {"post": {"summary": "Extract subgraph by seed node",
                "operationId": "get_kg_subgraph",
                "parameters": [{"name": "kg_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SubgraphRequest"}}}},
                "responses": {"200": {"description": "Subgraph JSON"}, "404": {"description": "Not found"}}}},
        },
        "components": {"schemas": {
            "KgJobRequest": {
                "type": "object",
                "required": ["run_id"],
                "properties": {
                    "run_id": {"type": "string"},
                    "max_entities": {"type": "integer", "default": 500},
                }
            },
            "SubgraphRequest": {
                "type": "object",
                "required": ["seed_node"],
                "properties": {
                    "seed_node": {"type": "string"},
                    "depth": {"type": "integer", "default": 2},
                    "max_nodes": {"type": "integer", "default": 50},
                }
            },
        }},
    },
    "ws": {
        "title": "ws-service",
        "version": "0.1.0",
        "description": "WebSocket gateway for real-time pipeline event streaming.",
        "paths": {
            "/health": {"get": {"summary": "Health check", "operationId": "health_get",
                "responses": {"200": {"description": "OK"}}}},
            "/": {"get": {"summary": "Service info & topic list", "operationId": "root_get",
                "responses": {"200": {"description": "Service metadata"}}}},
            "/ui/events": {"get": {
                "summary": "WebSocket event stream (upgrade required)",
                "operationId": "ui_events_ws",
                "description": "WebSocket endpoint. Send `Upgrade: websocket` header. Query param `topics` is a comma-separated list of pipeline topics to subscribe to (e.g. `ingestion,processing,eval`).",
                "parameters": [
                    {"name": "topics", "in": "query", "required": False,
                     "schema": {"type": "string"}, "description": "Comma-separated topic list"},
                ],
                "responses": {"101": {"description": "Switching Protocols — WebSocket upgrade"},
                              "400": {"description": "Unknown topic requested"}},
            }},
        },
        "components": {"schemas": {
            "Envelope": {
                "type": "object",
                "required": ["type", "seq", "ts"],
                "properties": {
                    "type": {"type": "string", "enum": ["welcome", "heartbeat", "data", "error"]},
                    "seq": {"type": "integer", "minimum": 0},
                    "ts": {"type": "number"},
                    "topic": {"type": "string"},
                    "payload": {"type": "object"},
                }
            }
        }},
    },
}


def generate_spec(name: str, spec: dict) -> dict:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": spec["title"],
            "version": spec["version"],
            "description": spec.get("description", ""),
        },
        "paths": spec["paths"],
        "components": spec.get("components", {"schemas": {}}),
    }


def main() -> int:
    root = pathlib.Path(__file__).parent.parent
    generated = []
    for svc_name, spec in SPECS.items():
        out_path = root / "services" / svc_name / "openapi.json"
        full_spec = generate_spec(svc_name, spec)
        with open(out_path, "w") as fh:
            json.dump(full_spec, fh, indent=2)
            fh.write("\n")
        print(f"  ✓  {out_path.relative_to(root)}")
        generated.append(str(out_path))
    print(f"\nGenerated {len(generated)} OpenAPI specs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
