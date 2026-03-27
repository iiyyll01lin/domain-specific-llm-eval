"""LangGraph node functions for the Agentic Self-Healing workflow.

Each function receives the current ``HealingState`` dict and returns a
partial dict of updated keys.  LangGraph merges the partial updates into
the running state automatically.

Node contract
-------------
* Nodes are pure functions (no side-effects other than logging) EXCEPT for
  ``engineer_node`` which writes to the ProposalStore staging table.
* ``commit_node`` is the **only** node that touches SQLiteGraphStore —
  it runs exclusively after human approval (enforced by ``interrupt_before``
  on the compiled graph).
* All heavy imports are done inside the function body so the module can be
  imported without requiring LangGraph or networkx in the test environment.
"""

from __future__ import annotations

import logging
import re
import uuid as _uuid
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SC_GAIN = 0.10  # minimum projected Sc improvement required to proceed
MAX_ENTITIES_PER_GAP = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_\u4e00-\u9fff]+", text.lower())


def _extract_key_terms(question: str, n: int = 4) -> List[str]:
    """Pull the n most discriminative tokens from a query string."""
    stop = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an",
            "in", "of", "to", "for", "with", "on", "at", "by", "from"}
    tokens = [t for t in _tokenize(question) if len(t) > 2 and t not in stop]
    # Deduplicate while preserving order
    seen: set = set()
    unique = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return unique[:n]


def _import_graph_store(db_path: str):
    """Lazily import SQLiteGraphStore so nodes work without eval-pipeline on path."""
    try:
        from src.utils.graph_store import SQLiteGraphStore, hash_content  # type: ignore[import]
        return SQLiteGraphStore(db_path), hash_content
    except ImportError:
        import sys
        from pathlib import Path
        _eval_dir = Path(__file__).resolve().parents[3] / "eval-pipeline"
        if str(_eval_dir) not in sys.path:
            sys.path.insert(0, str(_eval_dir))
        from src.utils.graph_store import SQLiteGraphStore, hash_content  # type: ignore[import]
        return SQLiteGraphStore(db_path), hash_content


# ---------------------------------------------------------------------------
# Node 1 — Diagnostician
# ---------------------------------------------------------------------------


def diagnose_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse the DriftResult and low-Sc queries to produce KnowledgeGaps.

    Gap detection heuristics
    ------------------------
    1. **disconnected_pair** — any query with Sc < 0.5 signals missing bridge nodes.
    2. **missing_entity**    — any query with Se < 0.4 signals absent domain entities.
    3. **hub_pollution**     — any query with Ph > 0.3 signals hub nodes inflating noise.
    4. **systematic_drift**  — drift result flags entity_overlap or structural_connectivity.

    The node is also used as the *re-diagnostician* when the verifier rejects
    a patch: it receives ``rejection_feedback`` and factors it into the description.
    """
    drift_result: Dict[str, Any] = state.get("drift_result") or {}
    low_sc_queries: List[Dict[str, Any]] = state.get("low_sc_queries") or []
    rejection_feedback: str = state.get("rejection_feedback") or ""

    gaps: List[Dict[str, Any]] = []

    for q in low_sc_queries:
        qid = q.get("query_id", "unknown")
        sc = float(q.get("structural_connectivity", 1.0))
        se = float(q.get("entity_overlap", 1.0))
        ph = float(q.get("hub_noise_penalty", 0.0))
        question = q.get("question", "")

        if sc < 0.5:
            gaps.append(
                {
                    "gap_type": "disconnected_pair",
                    "missing_entity": f"bridge_for_{qid}",
                    "affected_query_ids": [qid],
                    "severity_score": round(1.0 - sc, 4),
                    "description": (
                        f"Query '{question[:80]}' lacks graph connectivity (Sc={sc:.3f}). "
                        + (f"Previous repair rejected: {rejection_feedback}" if rejection_feedback else "")
                    ),
                }
            )

        if se < 0.4:
            for term in _extract_key_terms(question, MAX_ENTITIES_PER_GAP):
                gaps.append(
                    {
                        "gap_type": "missing_entity",
                        "missing_entity": term,
                        "affected_query_ids": [qid],
                        "severity_score": round(1.0 - se, 4),
                        "description": (
                            f"Entity '{term}' appears in query but is underrepresented "
                            f"in the KG (Se={se:.3f})."
                        ),
                    }
                )

        if ph > 0.3:
            gaps.append(
                {
                    "gap_type": "hub_pollution",
                    "missing_entity": f"non_hub_context_for_{qid}",
                    "affected_query_ids": [qid],
                    "severity_score": round(ph, 4),
                    "description": (
                        f"High hub-noise penalty (Ph={ph:.3f}) for query '{question[:60]}'. "
                        "New targeted nodes needed to dilute hub dominance."
                    ),
                }
            )

    # Systematic drift from DriftDetector
    for metric_key, meta in (drift_result.get("metrics") or {}).items():
        if not isinstance(meta, dict):
            continue
        if meta.get("flagged"):
            gaps.append(
                {
                    "gap_type": "systematic_drift",
                    "missing_entity": f"repair_{metric_key}",
                    "affected_query_ids": ["drift_triggered"],
                    "severity_score": min(1.0, abs(float(meta.get("delta_pct", 0.0))) / 100.0),
                    "description": (
                        f"Metric '{metric_key}' drifted {meta.get('delta_pct', 0):.1f}% "
                        f"from baseline (z={meta.get('z_score', 0):.2f})."
                    ),
                }
            )

    # Deduplicate by (gap_type, missing_entity)
    seen_keys: set = set()
    dedup: List[Dict[str, Any]] = []
    for g in gaps:
        key = (g["gap_type"], g["missing_entity"])
        if key not in seen_keys:
            seen_keys.add(key)
            dedup.append(g)

    dedup.sort(key=lambda g: -g["severity_score"])
    logger.info("Diagnostician: identified %d knowledge gaps.", len(dedup))

    return {
        "knowledge_gaps": dedup,
        "status": "RESEARCHING",
    }


# ---------------------------------------------------------------------------
# Node 2 — Web/SQL Researcher (internal mock DB only)
# ---------------------------------------------------------------------------


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Query the internal MockManufacturingDB to fill each KnowledgeGap.

    Retrieval is capped at 2 passages per gap to avoid flooding the Engineer
    with low-quality material.
    """
    from services.eval.agentic_healing.tools.db_researcher import MockManufacturingDB  # type: ignore[import]

    gaps: List[Dict[str, Any]] = state.get("knowledge_gaps") or []
    if not gaps:
        logger.warning("Researcher: no gaps to research — returning empty contexts.")
        return {"retrieved_contexts": [], "status": "ENGINEERING"}

    db = MockManufacturingDB()
    retrieved: List[Dict[str, Any]] = []

    for gap in gaps:
        entity = gap.get("missing_entity", "")
        description = gap.get("description", "")
        keywords = f"{entity} {description}"
        results = db.query(keywords=keywords, entity_type=gap.get("gap_type", ""), top_k=2)
        for ctx in results:
            retrieved.append({**ctx, "gap_reference": entity})

    logger.info("Researcher: retrieved %d contexts for %d gaps.", len(retrieved), len(gaps))
    return {
        "retrieved_contexts": retrieved,
        "status": "ENGINEERING",
    }


# ---------------------------------------------------------------------------
# Node 3 — Graph Engineer
# ---------------------------------------------------------------------------


def engineer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert retrieved contexts into a ProposedPatch, staged in ProposalStore.

    Safety guarantees
    -----------------
    * ``filter_new_hashes()`` ensures we propose ONLY nodes not already in the
      live graph — zero risk of duplicate proposals.
    * No ``upsert_node`` calls here; writes go exclusively to the staging table.
    * Edges are proposed only between nodes within the current patch (not to
      arbitrary existing nodes), preventing accidental graph rewiring.
    """
    from services.eval.agentic_healing.staging import ProposalStore  # type: ignore[import]

    db_path: str = state.get("db_path", "")
    contexts: List[Dict[str, Any]] = state.get("retrieved_contexts") or []
    gaps: List[Dict[str, Any]] = state.get("knowledge_gaps") or []

    store, hash_content = _import_graph_store(db_path)

    proposed_nodes: List[Dict[str, Any]] = []
    candidate_hashes: List[str] = []

    for ctx in contexts:
        content = ctx.get("content", "")
        if not content:
            continue
        h = hash_content(content)
        candidate_hashes.append(h)

    # Only propose genuinely new nodes
    new_hashes = set(store.filter_new_hashes(candidate_hashes)) if candidate_hashes else set()

    for ctx in contexts:
        content = ctx.get("content", "")
        if not content:
            continue
        h = hash_content(content)
        if h not in new_hashes:
            continue
        proposed_nodes.append(
            {
                "node_hash": h,
                "node_type": "document",
                "properties": {
                    "content": content,
                    "source_uri": ctx.get("source_uri", ""),
                    "entities": ctx.get("supporting_entities", []),
                    "keyphrases": ctx.get("keyphrases", []),
                    "confidence": ctx.get("confidence", 0.0),
                },
                "source_uri": ctx.get("source_uri", ""),
                "rationale": f"Resolves gap: {ctx.get('gap_reference', 'unknown')}",
            }
        )

    # Propose edges between co-proposed nodes that share entities
    proposed_edges: List[Dict[str, Any]] = []
    for i, n1 in enumerate(proposed_nodes):
        for n2 in proposed_nodes[i + 1:]:
            shared = set(n1["properties"]["entities"]) & set(n2["properties"]["entities"])
            if shared:
                proposed_edges.append(
                    {
                        "src_hash": n1["node_hash"],
                        "tgt_hash": n2["node_hash"],
                        "rel_type": "ENTITY_OVERLAP",
                        "properties": {"shared_entities": sorted(shared)[:5]},
                        "rationale": f"Shared entities: {', '.join(sorted(shared)[:3])}",
                    }
                )

    proposal_id = str(_uuid.uuid4())
    proposed_patch = {
        "proposal_id": proposal_id,
        "proposed_nodes": proposed_nodes,
        "proposed_edges": proposed_edges,
        "rationale": (
            f"Agentic repair: {len(gaps)} gaps identified, "
            f"{len(proposed_nodes)} new nodes, {len(proposed_edges)} new edges proposed."
        ),
        "estimated_sc_delta": 0.0,  # Updated by verify_node
    }

    # Write to staging — DOES NOT touch SQLiteGraphStore
    staging_db = str(state.get("db_path", "outputs/proposals.db")).replace(".db", "_proposals.db")
    try:
        proposal_store = ProposalStore(staging_db)
        proposal_store.write_proposal(
            proposal_id=proposal_id,
            thread_id=state.get("_thread_id", proposal_id),
            state={**state, "proposed_patch": proposed_patch, "proposal_id": proposal_id},
        )
    except Exception as exc:
        logger.error("ProposalStore write failed: %s", exc)

    logger.info(
        "Graph Engineer: staged proposal %s (%d nodes, %d edges).",
        proposal_id,
        len(proposed_nodes),
        len(proposed_edges),
    )

    return {
        "proposed_patch": proposed_patch,
        "proposal_id": proposal_id,
        "status": "VERIFYING",
    }


# ---------------------------------------------------------------------------
# Node 4 — Verifier (re-uses real Sc math, not LLM estimation)
# ---------------------------------------------------------------------------


def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate Sc on the augmented graph and verify that MIN_SC_GAIN is met.

    Algorithm
    ---------
    1. Load the full current graph from SQLiteGraphStore into a NetworkX object.
    2. Add proposed nodes + edges in-memory (no DB writes).
    3. For every low-Sc query, estimate the new Sc by computing the ratio of the
       largest connected component of the query's *simulated* subgraph.
    4. Compare mean ΔSc to MIN_SC_GAIN.

    This is deterministic arithmetic — no LLM estimation is used.
    """
    import networkx as nx  # type: ignore[import]

    proposed_patch = state.get("proposed_patch") or {}
    db_path: str = state.get("db_path", "")
    low_sc_queries: List[Dict[str, Any]] = state.get("low_sc_queries") or []
    iteration_count: int = int(state.get("iteration_count", 0))

    if not proposed_patch.get("proposed_nodes"):
        sc_delta = 0.0
        passed = False
        report = "FAILED: patch contains no proposed nodes — nothing to verify."
        return {
            "projected_sc_delta": sc_delta,
            "verification_report": report,
            "proposed_patch": proposed_patch,
            "iteration_count": iteration_count + 1,
        }

    try:
        store, _ = _import_graph_store(db_path)
        all_nodes = {n["node_hash"] for n in store.get_all_nodes()}
        all_rels = store.get_all_relationships()
    except Exception as exc:
        logger.warning("Could not load graph for Sc simulation: %s", exc)
        all_nodes = set()
        all_rels = []

    # Build the full current graph in-memory
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for rel in all_rels:
        G.add_edge(rel["src_hash"], rel["tgt_hash"])

    # Augment with proposed elements
    for node in proposed_patch["proposed_nodes"]:
        G.add_node(node["node_hash"])
    for edge in proposed_patch["proposed_edges"]:
        G.add_edge(edge["src_hash"], edge["tgt_hash"])

    # Compute projected Sc gains per low-Sc query
    proposed_hashes = {n["node_hash"] for n in proposed_patch["proposed_nodes"]}
    projected_gains: List[float] = []

    for q in low_sc_queries:
        old_sc = float(q.get("structural_connectivity", 0.0))

        # Approximate the "retrieved set" as a neighbourhood of proposed nodes
        # that share entity tokens with the query
        q_tokens = set(_tokenize(q.get("question", "")))
        relevant_new: List[str] = []
        for node in proposed_patch["proposed_nodes"]:
            node_tokens = set()
            for fld in ("entities", "keyphrases"):
                for v in node.get("properties", {}).get(fld, []):
                    node_tokens.update(_tokenize(str(v)))
            if q_tokens & node_tokens:
                relevant_new.append(node["node_hash"])

        if not relevant_new:
            # Fallback: assume all proposed nodes benefit this query equally
            relevant_new = list(proposed_hashes)

        # Build a simulated subgraph: existing neighbourhood + proposed nodes
        simulated_retrieved = set(relevant_new)
        # Add neighbours of proposed nodes from full graph (1-hop)
        for h in relevant_new:
            if G.has_node(h):
                simulated_retrieved.update(G.neighbors(h))

        subgraph = G.subgraph(simulated_retrieved)
        if len(simulated_retrieved) == 0:
            new_sc = old_sc
        else:
            components = list(nx.connected_components(subgraph))
            largest = max(len(c) for c in components) if components else 0
            new_sc = largest / len(simulated_retrieved)

        projected_gains.append(max(0.0, new_sc - old_sc))

    sc_delta = (
        sum(projected_gains) / len(projected_gains) if projected_gains else 0.0
    )
    sc_delta = round(sc_delta, 4)

    # Update patch with the computed delta
    proposed_patch = {**proposed_patch, "estimated_sc_delta": sc_delta}

    passed = sc_delta >= MIN_SC_GAIN
    report = (
        f"{'PASSED' if passed else 'FAILED'}: projected mean ΔSc={sc_delta:.4f} "
        f"(required≥{MIN_SC_GAIN}), "
        f"proposed_nodes={len(proposed_patch['proposed_nodes'])}, "
        f"iteration={iteration_count + 1}"
    )
    logger.info("Verifier: %s", report)

    # Increment iteration count only on failure
    new_iteration_count = iteration_count + (0 if passed else 1)

    return {
        "projected_sc_delta": sc_delta,
        "proposed_patch": proposed_patch,
        "verification_report": report,
        "iteration_count": new_iteration_count,
    }


# ---------------------------------------------------------------------------
# Node 5 — Commit (runs ONLY after human_approved=True via interrupt)
# ---------------------------------------------------------------------------


def commit_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the approved patch to the live SQLiteGraphStore.

    This is the **sole** node that calls ``upsert_node`` / ``add_relationship``.
    It can only run after a human has set ``human_approved=True`` on the
    insights-portal — enforced by the LangGraph ``interrupt_before`` mechanism.

    Idempotent: ``upsert_node`` on an existing hash is a no-op update; the
    ``UNIQUE`` constraint on ``relationships`` handles duplicate edge attempts.
    """
    from services.eval.agentic_healing.staging import ProposalStore  # type: ignore[import]

    if not state.get("human_approved"):
        # Should never happen due to interrupt_before, but guard defensively.
        logger.error("commit_node reached without human approval — aborting.")
        return {"status": "ABORTED", "abort_reason": "Missing human approval token."}

    proposed_patch = state.get("proposed_patch") or {}
    db_path: str = state.get("db_path", "")
    proposal_id: str = state.get("proposal_id", "")

    store, _ = _import_graph_store(db_path)

    committed_nodes = 0
    committed_edges = 0

    for node in proposed_patch.get("proposed_nodes", []):
        try:
            is_new = store.upsert_node(
                node["node_hash"], node["node_type"], node["properties"]
            )
            committed_nodes += 1
            logger.debug("Committed node %s (new=%s)", node["node_hash"][:8], is_new)
        except Exception as exc:
            logger.error("Failed to commit node %s: %s", node.get("node_hash", "?"), exc)

    for edge in proposed_patch.get("proposed_edges", []):
        try:
            store.add_relationship(
                edge["src_hash"], edge["tgt_hash"], edge["rel_type"], edge["properties"]
            )
            committed_edges += 1
        except Exception as exc:
            logger.error("Failed to commit edge %s→%s: %s", edge.get("src_hash", "?")[:8], edge.get("tgt_hash", "?")[:8], exc)

    # Update staging status
    staging_db = db_path.replace(".db", "_proposals.db") if db_path else "outputs/proposals.db"
    try:
        ProposalStore(staging_db).update_status(proposal_id, "committed")
    except Exception as exc:
        logger.warning("Could not update proposal status: %s", exc)

    logger.info(
        "Commit complete — %d nodes, %d edges applied to %s.",
        committed_nodes, committed_edges, db_path,
    )

    return {
        "status": "COMMITTED",
    }


# ---------------------------------------------------------------------------
# Node 6 — Abort (terminal node when MAX_ITERATIONS exceeded)
# ---------------------------------------------------------------------------


def abort_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Record the abort reason and terminate the workflow cleanly."""
    from services.eval.agentic_healing.staging import ProposalStore  # type: ignore[import]

    proposal_id: str = state.get("proposal_id", "")
    iteration_count: int = int(state.get("iteration_count", 0))
    best_delta: float = float(state.get("projected_sc_delta") or 0.0)
    reason = (
        f"Aborted after {iteration_count} iterations. "
        f"Best projected ΔSc={best_delta:.4f} never reached threshold {MIN_SC_GAIN}."
    )
    logger.warning("Abort: %s", reason)

    if proposal_id:
        db_path: str = state.get("db_path", "outputs/knowledge_graph.db")
        staging_db = db_path.replace(".db", "_proposals.db")
        try:
            ProposalStore(staging_db).update_status(proposal_id, "expired")
        except Exception as exc:
            logger.warning("Could not mark proposal as expired: %s", exc)

    return {
        "status": "ABORTED",
        "abort_reason": reason,
    }


# ---------------------------------------------------------------------------
# Routing function (used by conditional edge in graph.py)
# ---------------------------------------------------------------------------


def route_after_verify(state: Dict[str, Any]) -> str:
    """Select the next node based on verification outcome and iteration budget."""
    delta = float(state.get("projected_sc_delta") or 0.0)
    iteration = int(state.get("iteration_count", 0))
    max_iter = int(state.get("max_iterations", 3))

    if iteration >= max_iter:
        return "abort"
    if delta >= MIN_SC_GAIN:
        return "commit"  # will be interrupted by interrupt_before
    return "research"
