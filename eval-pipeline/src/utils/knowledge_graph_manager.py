#!/usr/bin/env python3
"""Knowledge graph persistence and reuse helpers for the evaluation pipeline."""

from __future__ import annotations

import json
import logging
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, TypedDict

from src.utils.graph_store import GraphStore, SQLiteGraphStore, hash_content

logger = logging.getLogger(__name__)


class KnowledgeGraphMetadata(TypedDict, total=False):
    run_id: str
    created_at: str
    nodes_count: int
    relationships_count: int
    source_documents: int
    notes: str


class KnowledgeGraphArtifact(TypedDict, total=False):
    json_path: str
    pickle_path: str
    created_at: str
    run_id: str
    nodes_count: int
    relationships_count: int
    file_size: int
    metadata: KnowledgeGraphMetadata


class KnowledgeGraphManager:
    """Manages knowledge graph storage, lookup, loading, and tenant filtering.

    Parameters
    ----------
    base_output_dir:
        Root directory for persisted artefacts (JSON snapshots, pickles).
    graph_store:
        Optional :class:`~src.utils.graph_store.GraphStore` backend.  When
        provided, every call to :meth:`save_knowledge_graph` will also sync the
        nodes and relationships into the persistent store, enabling incremental
        re-use across runs.  If omitted, behaviour is identical to the legacy
        flat-file implementation.
    """

    def __init__(
        self,
        base_output_dir: str | Path,
        graph_store: Optional[GraphStore] = None,
    ) -> None:
        self.base_output_dir = Path(base_output_dir)
        self.testset_kg_dir = self.base_output_dir / "testsets" / "knowledge_graphs"
        self.management_kg_dir = self.base_output_dir / "metadata" / "knowledge_graphs"
        self.testset_kg_dir.mkdir(parents=True, exist_ok=True)
        self.management_kg_dir.mkdir(parents=True, exist_ok=True)
        self.kg: Optional[Any] = None
        self.graph_store: Optional[GraphStore] = graph_store

    def save_knowledge_graph(
        self,
        kg: Any,
        metadata: Optional[KnowledgeGraphMetadata] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Persist a KG as JSON for RAGAS reuse and as pickle for local reuse."""
        resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        serialized = self._serialize_knowledge_graph(kg)

        artifact_metadata: KnowledgeGraphMetadata = {
            "run_id": resolved_run_id,
            "created_at": datetime.now().isoformat(),
            "nodes_count": len(serialized.get("nodes", [])),
            "relationships_count": len(serialized.get("relationships", [])),
        }
        if metadata:
            artifact_metadata.update(metadata)
        serialized["metadata"] = artifact_metadata

        json_path = self.testset_kg_dir / f"knowledge_graph_{resolved_run_id}.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(serialized, handle, indent=2, ensure_ascii=False, default=str)

        pickle_path = self.management_kg_dir / f"knowledge_graph_{resolved_run_id}.pkl"
        with open(pickle_path, "wb") as handle:
            pickle.dump(kg, handle)

        self.kg = kg
        logger.info("💾 Knowledge graph saved to %s", json_path)
        logger.info("💾 Knowledge graph pickle saved to %s", pickle_path)

        # --- Sync into the persistent GraphStore (incremental, deduplicating) ---
        if self.graph_store is not None:
            self._sync_to_graph_store(kg)

        return str(json_path)

    def list_available_knowledge_graphs(self) -> List[KnowledgeGraphArtifact]:
        """List all persisted JSON knowledge graph artifacts sorted by newest first."""
        artifacts: List[KnowledgeGraphArtifact] = []
        for json_path in sorted(self.testset_kg_dir.glob("knowledge_graph_*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                metadata = payload.get("metadata", {})
                run_id = str(metadata.get("run_id", json_path.stem.replace("knowledge_graph_", "")))
                pickle_path = self.management_kg_dir / f"knowledge_graph_{run_id}.pkl"
                artifacts.append(
                    {
                        "json_path": str(json_path),
                        "pickle_path": str(pickle_path) if pickle_path.exists() else "",
                        "created_at": str(metadata.get("created_at", datetime.fromtimestamp(json_path.stat().st_mtime).isoformat())),
                        "run_id": run_id,
                        "nodes_count": int(metadata.get("nodes_count", len(payload.get("nodes", [])))),
                        "relationships_count": int(metadata.get("relationships_count", len(payload.get("relationships", [])))),
                        "file_size": json_path.stat().st_size,
                        "metadata": metadata,
                    }
                )
            except Exception as exc:
                logger.warning("⚠️ Skipping unreadable KG artifact %s: %s", json_path, exc)

        artifacts.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return artifacts

    def get_latest_knowledge_graph(self) -> Optional[str]:
        """Return the newest persisted JSON knowledge graph path if present."""
        artifacts = self.list_available_knowledge_graphs()
        if not artifacts:
            return None
        return artifacts[0].get("json_path")

    def load_knowledge_graph_json(self, json_path: str | Path) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_knowledge_graph_pickle(self, pickle_path: str | Path) -> Any:
        with open(pickle_path, "rb") as handle:
            kg = pickle.load(handle)
        self.kg = kg
        return kg

    def apply_tenant_isolation(self, tenant_id: str, role: str) -> None:
        """Filter the loaded KG in-place based on simple tenant and role metadata."""
        if self.kg is None or not hasattr(self.kg, "nodes"):
            return

        role_rank = {"viewer": 0, "editor": 1, "admin": 2}
        current_role_rank = role_rank.get(role, 0)

        allowed_nodes = []
        allowed_node_ids = set()
        for node in list(getattr(self.kg, "nodes", [])):
            properties = getattr(node, "properties", {}) or {}
            node_tenant = str(properties.get("tenant", "public"))
            node_role = str(properties.get("min_role", "viewer"))
            if node_tenant not in {"public", tenant_id}:
                continue
            if current_role_rank < role_rank.get(node_role, 0):
                continue
            allowed_nodes.append(node)
            node_id = getattr(node, "id", None)
            if node_id is not None:
                allowed_node_ids.add(str(node_id))

        if hasattr(self.kg, "relationships"):
            filtered_relationships = []
            for relationship in list(getattr(self.kg, "relationships", [])):
                source_id = self._extract_endpoint_id(getattr(relationship, "source", None))
                target_id = self._extract_endpoint_id(getattr(relationship, "target", None))
                if source_id in allowed_node_ids and target_id in allowed_node_ids:
                    filtered_relationships.append(relationship)
            self.kg.relationships = filtered_relationships

        self.kg.nodes = allowed_nodes
        logger.info(
            "🔒 Tenant isolation applied for %s/%s. Kept %s nodes.",
            tenant_id,
            role,
            len(allowed_nodes),
        )

    def _sync_to_graph_store(self, kg: Any) -> None:
        """Sync nodes and relationships from a RAGAS KG into the GraphStore.

        Uses the node's ``content`` property as the canonical text for
        content-hash addressing.  Falls back to the node ``id`` string when
        ``content`` is absent.
        """
        if self.graph_store is None:
            return

        node_id_to_hash: Dict[str, str] = {}
        nodes_synced = 0
        for node in list(getattr(kg, "nodes", [])):
            properties = deepcopy(getattr(node, "properties", {}) or {})
            content = str(properties.get("content") or getattr(node, "id", ""))
            node_hash = hash_content(content)
            node_id_to_hash[str(getattr(node, "id", ""))] = node_hash

            # Preserve RAGAS node UUID for stable round-trip reconstruction
            if "_node_uuid" not in properties:
                properties["_node_uuid"] = str(getattr(node, "id", ""))

            self.graph_store.upsert_node(
                node_hash=node_hash,
                node_type=str(getattr(node, "type", "document")),
                properties=properties,
            )
            nodes_synced += 1

        rels_synced = 0
        for rel in list(getattr(kg, "relationships", []) or []):
            src_id = self._extract_endpoint_id(getattr(rel, "source", None))
            tgt_id = self._extract_endpoint_id(getattr(rel, "target", None))
            src_hash = node_id_to_hash.get(src_id)
            tgt_hash = node_id_to_hash.get(tgt_id)
            if src_hash and tgt_hash:
                self.graph_store.add_relationship(
                    src_hash=src_hash,
                    tgt_hash=tgt_hash,
                    rel_type=str(getattr(rel, "type", getattr(rel, "relation_type", "related_to"))),
                    properties=deepcopy(getattr(rel, "properties", {}) or {}),
                )
                rels_synced += 1

        logger.info(
            "\ud83d\udd04 Synced %d nodes and %d relationships to GraphStore.",
            nodes_synced,
            rels_synced,
        )

    def _serialize_knowledge_graph(self, kg: Any) -> Dict[str, Any]:
        nodes = []
        for node in list(getattr(kg, "nodes", [])):
            nodes.append(
                {
                    "id": str(getattr(node, "id", "")),
                    "label": str(getattr(node, "label", "") or getattr(node, "name", "")),
                    "type": str(getattr(node, "type", "entity")),
                    "properties": deepcopy(getattr(node, "properties", {}) or {}),
                }
            )

        relationships = []
        for relationship in list(getattr(kg, "relationships", [])):
            relationships.append(
                {
                    "id": str(getattr(relationship, "id", "")),
                    "source": self._extract_endpoint_id(getattr(relationship, "source", None)),
                    "target": self._extract_endpoint_id(getattr(relationship, "target", None)),
                    "type": str(getattr(relationship, "type", getattr(relationship, "relation_type", "related_to"))),
                    "properties": deepcopy(getattr(relationship, "properties", {}) or {}),
                }
            )

        return {"nodes": nodes, "relationships": relationships}

    @staticmethod
    def _extract_endpoint_id(endpoint: Any) -> str:
        if endpoint is None:
            return ""
        if hasattr(endpoint, "id"):
            return str(getattr(endpoint, "id"))
        return str(endpoint)


def find_and_use_latest_kg(
    config: MutableMapping[str, Any],
    output_dir: str | Path,
) -> MutableMapping[str, Any]:
    """Update config to point at the newest persisted knowledge graph if available."""
    kg_manager = KnowledgeGraphManager(output_dir)
    latest_kg = kg_manager.get_latest_knowledge_graph()
    if latest_kg is None:
        return config

    testset_generation = config.setdefault("testset_generation", {})
    ragas_config = testset_generation.setdefault("ragas_config", {})
    knowledge_graph_config = ragas_config.setdefault("knowledge_graph_config", {})
    knowledge_graph_config["existing_kg_file"] = latest_kg
    logger.info("🔁 Configured latest knowledge graph reuse: %s", latest_kg)
    return config


__all__ = ["KnowledgeGraphManager", "KnowledgeGraphArtifact", "KnowledgeGraphMetadata", "find_and_use_latest_kg"]
