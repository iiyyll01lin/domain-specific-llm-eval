"""Type definitions for the Agentic Self-Healing Knowledge Graph workflow.

All dataclasses serialise to / deserialise from plain ``dict`` so they can be
stored in SQLite (staging) and transmitted over JSON (FastAPI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class HealingStatus(str, Enum):
    DIAGNOSING = "DIAGNOSING"
    RESEARCHING = "RESEARCHING"
    ENGINEERING = "ENGINEERING"
    VERIFYING = "VERIFYING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


# ---------------------------------------------------------------------------
# Input types (trigger data)
# ---------------------------------------------------------------------------


@dataclass
class LowScoreQuery:
    """A single RAG query that scored below the critical threshold."""

    query_id: str
    question: str
    entity_overlap: float
    structural_connectivity: float
    hub_noise_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "entity_overlap": self.entity_overlap,
            "structural_connectivity": self.structural_connectivity,
            "hub_noise_penalty": self.hub_noise_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LowScoreQuery":
        return cls(
            query_id=d["query_id"],
            question=d.get("question", ""),
            entity_overlap=float(d.get("entity_overlap", 0.0)),
            structural_connectivity=float(d.get("structural_connectivity", 0.0)),
            hub_noise_penalty=float(d.get("hub_noise_penalty", 0.0)),
        )


# ---------------------------------------------------------------------------
# Diagnostician output
# ---------------------------------------------------------------------------

GapType = str  # "missing_entity" | "disconnected_pair" | "hub_pollution" | "systematic_drift"


@dataclass
class KnowledgeGap:
    """A single identified deficiency in the Knowledge Graph."""

    gap_type: GapType
    missing_entity: str
    affected_query_ids: List[str]
    severity_score: float  # 0.0–1.0, higher = more urgent
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "missing_entity": self.missing_entity,
            "affected_query_ids": self.affected_query_ids,
            "severity_score": self.severity_score,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeGap":
        return cls(
            gap_type=d["gap_type"],
            missing_entity=d["missing_entity"],
            affected_query_ids=d.get("affected_query_ids", []),
            severity_score=float(d.get("severity_score", 0.0)),
            description=d.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Researcher output
# ---------------------------------------------------------------------------


@dataclass
class RetrievedContext:
    """A context chunk retrieved by the Researcher for a specific gap."""

    content: str
    source_uri: str
    confidence: float
    supporting_entities: List[str]
    keyphrases: List[str]
    gap_reference: str  # which KnowledgeGap.missing_entity this context resolves

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_uri": self.source_uri,
            "confidence": self.confidence,
            "supporting_entities": self.supporting_entities,
            "keyphrases": self.keyphrases,
            "gap_reference": self.gap_reference,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievedContext":
        return cls(
            content=d["content"],
            source_uri=d.get("source_uri", "internal://mock_db"),
            confidence=float(d.get("confidence", 0.5)),
            supporting_entities=d.get("supporting_entities", []),
            keyphrases=d.get("keyphrases", []),
            gap_reference=d.get("gap_reference", ""),
        )


# ---------------------------------------------------------------------------
# Graph Engineer output (proposed patch — never auto-committed)
# ---------------------------------------------------------------------------


@dataclass
class ProposedNode:
    node_hash: str
    node_type: str
    properties: Dict[str, Any]
    source_uri: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_hash": self.node_hash,
            "node_type": self.node_type,
            "properties": self.properties,
            "source_uri": self.source_uri,
            "rationale": self.rationale,
        }


@dataclass
class ProposedEdge:
    src_hash: str
    tgt_hash: str
    rel_type: str
    properties: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_hash": self.src_hash,
            "tgt_hash": self.tgt_hash,
            "rel_type": self.rel_type,
            "properties": self.properties,
            "rationale": self.rationale,
        }


@dataclass
class ProposedPatch:
    """A staged, human-approvable graph repair proposal."""

    proposal_id: str
    proposed_nodes: List[ProposedNode] = field(default_factory=list)
    proposed_edges: List[ProposedEdge] = field(default_factory=list)
    rationale: str = ""
    estimated_sc_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposed_nodes": [n.to_dict() for n in self.proposed_nodes],
            "proposed_edges": [e.to_dict() for e in self.proposed_edges],
            "rationale": self.rationale,
            "estimated_sc_delta": self.estimated_sc_delta,
        }


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class HealingState(dict):  # type: ignore[type-arg]
    """Mutable workflow state threaded through every LangGraph node.

    Typed as ``dict`` subclass so LangGraph can serialise/deserialise it via
    its standard checkpointer without requiring TypedDict-only constraints.
    All keys are optional (total=False equivalent) — nodes update only the keys
    they own.
    """

    # Keys (documented for IDE support):
    #   drift_result       : Dict[str, Any]       trigger payload from DriftDetector
    #   low_sc_queries     : List[Dict]            LowScoreQuery.to_dict() entries
    #   knowledge_gaps     : List[Dict]            KnowledgeGap.to_dict() entries
    #   retrieved_contexts : List[Dict]            RetrievedContext.to_dict() entries
    #   proposed_patch     : Optional[Dict]        ProposedPatch.to_dict() or None
    #   proposal_id        : Optional[str]
    #   verification_report: Optional[str]
    #   projected_sc_delta : Optional[float]
    #   human_approved     : Optional[bool]        None=pending
    #   rejection_feedback : Optional[str]
    #   iteration_count    : int
    #   max_iterations     : int
    #   status             : str                   HealingStatus value
    #   db_path            : str                   SQLiteGraphStore file path
    #   abort_reason       : Optional[str]
