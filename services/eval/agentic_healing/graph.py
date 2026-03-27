"""LangGraph StateGraph definition for the Agentic Self-Healing workflow.

Graph topology
--------------

    START ──► diagnose ──► research ──► engineer ──► verify
                                           ▲              │
                             (re-research) │   route_after_verify
                                           └──────────────┤
                                                          │ pass  → commit ──► END
                                                          │ abort → abort  ──► END

``commit`` has ``interrupt_before`` set so the graph **pauses** before that
node and persists its checkpoint.  The FastAPI approve endpoint resumes it by
calling ``graph.invoke(None, config={"configurable": {"thread_id": ...}})``.

Usage
-----
>>> from services.eval.agentic_healing.graph import build_healing_graph
>>> graph = build_healing_graph("outputs/healing_checkpoints.db")
>>> config = {"configurable": {"thread_id": "run-abc123"}}
>>> result = graph.invoke(initial_state, config=config)
>>> # ── graph pauses here at the interrupt ──
>>> graph.update_state(config, {"human_approved": True})
>>> result = graph.invoke(None, config=config)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_healing_graph(checkpointer_db_path: str) -> Any:
    """Build and return a compiled LangGraph ``CompiledStateGraph``.

    Parameters
    ----------
    checkpointer_db_path:
        Filesystem path for the LangGraph SQLite checkpointer database.
        The parent directory is created automatically.

    Returns
    -------
    CompiledStateGraph
        A LangGraph graph compiled with a ``SqliteSaver`` checkpointer and
        ``interrupt_before=["commit"]``.
    """
    from pathlib import Path

    from langgraph.graph import StateGraph, END  # type: ignore[import]
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

    from services.eval.agentic_healing.nodes import (  # type: ignore[import]
        diagnose_node,
        research_node,
        engineer_node,
        verify_node,
        commit_node,
        abort_node,
        route_after_verify,
    )

    Path(checkpointer_db_path).parent.mkdir(parents=True, exist_ok=True)

    builder: StateGraph = StateGraph(dict)

    # ── Register nodes ──────────────────────────────────────────────────────
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("research", research_node)
    builder.add_node("engineer", engineer_node)
    builder.add_node("verify", verify_node)
    builder.add_node("commit", commit_node)
    builder.add_node("abort", abort_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    builder.set_entry_point("diagnose")

    # ── Linear edges ────────────────────────────────────────────────────────
    builder.add_edge("diagnose", "research")
    builder.add_edge("research", "engineer")
    builder.add_edge("engineer", "verify")

    # ── Conditional edge from verify ────────────────────────────────────────
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "research": "research",  # retry on low delta
            "commit": "commit",      # passes MIN_SC_GAIN → human approval gate
            "abort": "abort",        # MAX_ITERATIONS exceeded
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    builder.add_edge("commit", END)
    builder.add_edge("abort", END)

    # ── Compile with checkpointer and human-in-the-loop interrupt ───────────
    checkpointer = SqliteSaver.from_conn_string(checkpointer_db_path)
    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["commit"],
    )

    logger.info(
        "Agentic healing graph compiled — interrupt_before=['commit'], "
        "checkpointer=%s",
        checkpointer_db_path,
    )
    return compiled
