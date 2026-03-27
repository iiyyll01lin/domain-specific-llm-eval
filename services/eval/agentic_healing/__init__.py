"""Agentic Self-Healing Knowledge Graph Workflow.

This package implements an autonomous, LangGraph-powered multi-agent system
that detects Knowledge Graph deficiencies (triggered by DriftDetector) and
proposes—but never auto-commits—targeted repairs for human review.

Workflow stages
---------------
1. **diagnose**  — Analyse low-Sc queries and DriftResult to produce
                   a ranked list of KnowledgeGap objects.
2. **research**  — Query the internal mock manufacturing DB to retrieve
                   contexts that resolve each gap.
3. **engineer**  — Hash-address the retrieved contexts, deduplicate against
                   the live SQLiteGraphStore, and stage a ProposedPatch.
4. **verify**    — Re-run simulated Sc arithmetic on the augmented graph to
                   confirm the patch clears MIN_SC_GAIN (0.10).
                   Cycles back to **research** on failure; aborts after
                   MAX_ITERATIONS (3).
5. **commit**    — *interrupt_before* gate — resumes only when a human sets
                   ``human_approved = True`` via the insights-portal.

Public API
----------
``build_healing_graph(checkpointer_db_path)``
    Returns a compiled LangGraph ``CompiledStateGraph`` ready for invocation.

``HealingState``
    TypedDict consumed by every node and the FastAPI routes.
"""

from .graph import build_healing_graph
from .state import HealingState, HealingStatus

__all__ = ["build_healing_graph", "HealingState", "HealingStatus"]
