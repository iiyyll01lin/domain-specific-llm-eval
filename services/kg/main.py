"""KG Builder Service — FastAPI application.

TASK-060: Accept build requests, schedule extraction + relationship building,
expose status, summary, and subgraph endpoints.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from services.common.config import configure_service
from services.common.errors import (
    ServiceError,
    generic_error_handler,
    service_error_handler,
    validation_error_handler,
)
from services.common.middleware import TraceMiddleware
from services.kg.extract import extract_all
from services.kg.relationships import build_all_relationships
from services.kg.repository import KgJob, KgRepository
from services.kg.subgraph import sample_subgraph
from services.kg.summary import build_kg_summary

logger = logging.getLogger(__name__)

SERVICE_NAME = "kg-service"
configure_service(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME)
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


@lru_cache
def get_repository() -> KgRepository:
    from services.common.config import settings

    return KgRepository(settings.kg_db_path)


@lru_cache
def get_outputs_dir() -> Path:
    from services.common.config import settings

    p = Path(settings.kg_outputs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class KgDocument(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KgBuildRequest(BaseModel):
    documents: List[KgDocument]
    jaccard_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    overlap_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    cosine_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    summary_cosine_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_entities: int = Field(default=30, ge=1)
    max_keyphrases: int = Field(default=10, ge=1)


class KgSubgraphRequest(BaseModel):
    seed_entity: Optional[str] = None
    max_nodes: int = Field(default=500, ge=1, le=500)
    seed: int = Field(default=42)


# ---------------------------------------------------------------------------
# Background task: build KG
# ---------------------------------------------------------------------------


def _build_kg_background(
    kg_id: str,
    request: KgBuildRequest,
    repo: KgRepository,
    outputs_dir: Path,
) -> None:
    repo.update_status(kg_id, "running")
    try:
        nodes: List[Dict[str, Any]] = []
        for doc in request.documents:
            props = extract_all(
                doc.text,
                max_entities=request.max_entities,
                max_keyphrases=request.max_keyphrases,
            )
            nodes.append(
                {
                    "node_id": doc.doc_id,
                    "text": doc.text,
                    "entities": props["entities"],
                    "keyphrases": props["keyphrases"],
                    "sentences": props["sentences"],
                    "metadata": doc.metadata,
                }
            )

        relationships = build_all_relationships(
            nodes,
            jaccard_threshold=request.jaccard_threshold,
            overlap_threshold=request.overlap_threshold,
            cosine_threshold=request.cosine_threshold,
            summary_cosine_threshold=request.summary_cosine_threshold,
        )

        # Persist artefacts
        kg_dir = outputs_dir / kg_id
        kg_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = kg_dir / "nodes.json"
        rels_path = kg_dir / "relationships.json"
        nodes_path.write_text(json.dumps(nodes, ensure_ascii=False, indent=2))
        rels_path.write_text(json.dumps(relationships, ensure_ascii=False, indent=2))

        repo.update_completed(
            kg_id=kg_id,
            node_count=len(nodes),
            edge_count=len(relationships),
            artifacts={"nodes": str(nodes_path), "relationships": str(rels_path)},
        )
        logger.info("KG %s built: %d nodes, %d edges", kg_id, len(nodes), len(relationships))
    except Exception as exc:
        logger.exception("KG build failed for %s", kg_id)
        repo.update_failed(kg_id, str(exc))


# ---------------------------------------------------------------------------
# Helper: load artefacts from disk
# ---------------------------------------------------------------------------


def _load_kg_artefacts(job: KgJob, outputs_dir: Path):
    nodes_path = job.artifacts.get("nodes") or str(outputs_dir / job.kg_id / "nodes.json")
    rels_path = job.artifacts.get("relationships") or str(
        outputs_dir / job.kg_id / "relationships.json"
    )
    try:
        nodes = json.loads(Path(nodes_path).read_text())
        rels = json.loads(Path(rels_path).read_text())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"KG artefacts not found: {exc}") from exc
    return nodes, rels


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "kg", "message": "KG builder service"}


@app.post("/kg-jobs", status_code=status.HTTP_202_ACCEPTED)
async def submit_kg_job(
    request: KgBuildRequest,
    background_tasks: BackgroundTasks,
    repo: KgRepository = Depends(get_repository),
) -> Dict[str, Any]:
    if not request.documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="documents must not be empty",
        )
    job = repo.create(doc_count=len(request.documents))
    outputs_dir = get_outputs_dir()
    background_tasks.add_task(_build_kg_background, job.kg_id, request, repo, outputs_dir)
    return {
        "kg_id": job.kg_id,
        "status": job.status,
        "status_url": f"/kg-jobs/{job.kg_id}",
    }


@app.get("/kg-jobs")
async def list_kg_jobs(
    repo: KgRepository = Depends(get_repository),
) -> List[Dict[str, Any]]:
    jobs = repo.list_all()
    return [
        {
            "kg_id": j.kg_id,
            "status": j.status,
            "doc_count": j.doc_count,
            "node_count": j.node_count,
            "edge_count": j.edge_count,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
        }
        for j in jobs
    ]


@app.get("/kg-jobs/{kg_id}")
async def get_kg_job(
    kg_id: str,
    repo: KgRepository = Depends(get_repository),
) -> Dict[str, Any]:
    job = repo.get(kg_id)
    if job is None:
        raise HTTPException(status_code=404, detail="KG job not found")
    return {
        "kg_id": job.kg_id,
        "status": job.status,
        "doc_count": job.doc_count,
        "node_count": job.node_count,
        "edge_count": job.edge_count,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error_message": job.error_message,
    }


@app.get("/kg-jobs/{kg_id}/summary")
async def get_kg_summary(
    kg_id: str,
    top: int = 20,
    repo: KgRepository = Depends(get_repository),
) -> Dict[str, Any]:
    job = repo.get(kg_id)
    if job is None:
        raise HTTPException(status_code=404, detail="KG job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=409, detail=f"KG not ready — status: {job.status}"
        )
    outputs_dir = get_outputs_dir()
    nodes, rels = _load_kg_artefacts(job, outputs_dir)
    return build_kg_summary(nodes, rels, kg_id=kg_id, top_entity_limit=top)


@app.post("/kg-jobs/{kg_id}/subgraph")
async def get_subgraph(
    kg_id: str,
    request: KgSubgraphRequest,
    repo: KgRepository = Depends(get_repository),
) -> Dict[str, Any]:
    job = repo.get(kg_id)
    if job is None:
        raise HTTPException(status_code=404, detail="KG job not found")
    if job.status != "completed":
        raise HTTPException(
            status_code=409, detail=f"KG not ready — status: {job.status}"
        )
    outputs_dir = get_outputs_dir()
    nodes, rels = _load_kg_artefacts(job, outputs_dir)
    return sample_subgraph(
        nodes,
        rels,
        seed_entity=request.seed_entity,
        max_nodes=request.max_nodes,
        seed=request.seed,
    )
