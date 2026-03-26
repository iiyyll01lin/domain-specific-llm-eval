#!/usr/bin/env python3
"""Golden-Path E2E Runner.

A purpose-built, fully-offline CLI tool that exercises the complete
evaluation "golden path":

    CSV corpus  →  SQLiteGraphStore  →  EvaluationDispatcher (GraphSpec)
                →  GraphContextRelevanceEvaluator  →  JSON + CSV + XLSX artifacts

Usage::

    python scripts/golden_path_runner.py \\
        --corpus  tests/fixtures/golden_corpus.csv \\
        --config  tests/fixtures/golden_path_config.yaml \\
        --output-dir /workspace/outputs/golden_path

Exit codes:
    0  All assertions passed, artifacts written.
    1  An assertion failed (score == 0, missing contract key, etc.).
    2  An unexpected exception occurred.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path bootstrap — works whether the script is called from the repo root,
# eval-pipeline/, or /workspace/ inside Docker.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent        # eval-pipeline/scripts/
_PIPELINE_DIR = _SCRIPT_DIR.parent                   # eval-pipeline/
_REPO_ROOT = _PIPELINE_DIR.parent                    # repo root / /workspace

for _p in [
    _PIPELINE_DIR,
    _PIPELINE_DIR / "src",
    _REPO_ROOT / "ragas" / "ragas" / "src",
]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from src.evaluation.evaluation_dispatcher import EvaluationDispatcher, GraphSpec
from src.evaluation.graph_context_relevance import GraphContextRelevanceEvaluator
from src.utils.graph_store import SQLiteGraphStore, hash_content
from src.utils.pipeline_file_saver import PipelineFileSaver

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required keys that every valid evaluation contract must contain.
# Mirrors CONTRACT_KEYS used in test_graph_context_relevance.py.
# ---------------------------------------------------------------------------
REQUIRED_CONTRACT_KEYS = {
    "backend",
    "entity_overlap",
    "structural_connectivity",
    "hub_noise_penalty",
    "hub_nodes",
    "largest_component_size",
    "retrieved_count",
    "alpha",
    "beta",
    "gamma",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    """Read corpus CSV and return list of row dicts."""
    with open(corpus_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_store(
    db_path: Path,
    corpus_rows: List[Dict[str, str]],
    relationships: List[List[Any]],
) -> tuple[SQLiteGraphStore, List[str]]:
    """Create a clean SQLiteGraphStore populated with the golden corpus."""
    store = SQLiteGraphStore(db_path)
    hashes: List[str] = []

    for row in corpus_rows:
        content = row["content"]
        doc_type = row.get("doc_type", "document")
        h = hash_content(content)
        store.upsert_node(
            h,
            doc_type,
            {"content": content, "keyphrases": content},
        )
        hashes.append(h)

    for rel in relationships:
        src_i, tgt_i, rel_type, score = int(rel[0]), int(rel[1]), str(rel[2]), float(rel[3])
        store.add_relationship(
            hashes[src_i],
            hashes[tgt_i],
            rel_type,
            {"score": score},
        )

    return store, hashes


def _assert_result(result: Dict[str, Any], expected_count: int) -> None:
    """Raise AssertionError on any contract violation."""
    score = result.get("score")
    assert score is not None, "Result missing 'score' key"
    assert isinstance(score, float), f"score must be float, got {type(score).__name__}"
    assert 0.0 < score <= 1.0, (
        f"Expected 0 < score ≤ 1.0, got {score}. "
        "Check that corpus nodes share tokens with the query/answer."
    )

    contract = result.get("contract")
    assert contract is not None, "Result missing 'contract' key"

    missing_keys = REQUIRED_CONTRACT_KEYS - contract.keys()
    assert not missing_keys, f"Contract missing required keys: {missing_keys}"

    assert contract["backend"] == "graph_context_relevance", (
        f"Unexpected backend: {contract['backend']!r}"
    )
    assert contract["retrieved_count"] == expected_count, (
        f"Expected retrieved_count={expected_count}, got {contract['retrieved_count']}"
    )


def _write_artifacts(
    result: Dict[str, Any],
    output_dir: Path,
    query: str,
    expected_answer: str,
) -> Dict[str, str]:
    """Save JSON, CSV, and XLSX outputs; return mapping of type → path."""
    # 1. Raw evaluation result as JSON
    result_json = output_dir / "evaluation_result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Report row(s) via PipelineFileSaver → CSV in standardised layout
    saver = PipelineFileSaver(output_dir)
    samples = [
        {
            "user_input": query,
            "reference": expected_answer,
            "gcr_score": result["score"],
            "entity_overlap": result["contract"]["entity_overlap"],
            "structural_connectivity": result["contract"]["structural_connectivity"],
            "hub_noise_penalty": result["contract"]["hub_noise_penalty"],
            "retrieved_count": result["contract"]["retrieved_count"],
            "backend": result["contract"]["backend"],
        }
    ]
    csv_path = saver.save_testset_csv(samples, filename_prefix="golden_path_eval")

    # 3. Excel summary (openpyxl is in requirements.txt)
    xlsx_path = output_dir / "golden_path_report.xlsx"
    pd.DataFrame(samples).to_excel(str(xlsx_path), index=False, engine="openpyxl")

    return {
        "json": str(result_json),
        "csv": csv_path,
        "xlsx": str(xlsx_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Golden-Path E2E Runner — offline, deterministic pipeline smoke test"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to golden corpus CSV (columns: content, doc_type)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to golden-path config YAML",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output artifacts (created if absent)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 1: Load inputs ────────────────────────────────────────────
        corpus_rows = _load_corpus(Path(args.corpus))
        cfg = _load_config(Path(args.config))
        gc_cfg = cfg["golden_corpus"]
        eval_cfg = cfg.get("evaluation", {})

        print(f"📄 Loaded corpus: {len(corpus_rows)} documents from {args.corpus}")

        # ── Step 2: Build SQLiteGraphStore ────────────────────────────────
        db_path = output_dir / "golden_path.db"
        store, hashes = _build_store(db_path, corpus_rows, gc_cfg["relationships"])
        print(
            f"🧠 SQLiteGraphStore initialised: {len(hashes)} nodes, "
            f"{len(gc_cfg['relationships'])} relationships → {db_path}"
        )

        # ── Step 3: Wire evaluator and dispatcher ─────────────────────────
        graph_evaluator = GraphContextRelevanceEvaluator(
            store=store,
            alpha=float(eval_cfg.get("alpha", 0.4)),
            beta=float(eval_cfg.get("beta", 0.4)),
            gamma=float(eval_cfg.get("gamma", 0.2)),
        )
        # rag_evaluator=None is safe: only GraphSpec dispatch is exercised here.
        dispatcher = EvaluationDispatcher(
            rag_evaluator=None,
            graph_evaluator=graph_evaluator,
        )

        # ── Step 4: Dispatch GraphSpec ────────────────────────────────────
        spec = GraphSpec(
            question=gc_cfg["query"],
            expected_answer=gc_cfg["expected_answer"],
            retrieved_node_hashes=hashes,
        )
        result = dispatcher.dispatch(spec)
        print(
            f"✅ Evaluation dispatched: score={result['score']:.4f}  "
            f"entity_overlap={result['contract']['entity_overlap']:.4f}  "
            f"structural_connectivity={result['contract']['structural_connectivity']:.4f}"
        )

        # ── Step 5: Assert contract ───────────────────────────────────────
        _assert_result(result, expected_count=len(hashes))
        print("✅ All contract assertions passed")

        # ── Step 6: Write artifacts ───────────────────────────────────────
        artifacts = _write_artifacts(
            result, output_dir, gc_cfg["query"], gc_cfg["expected_answer"]
        )
        print(f"✅ JSON  → {artifacts['json']}")
        print(f"✅ CSV   → {artifacts['csv']}")
        print(f"✅ XLSX  → {artifacts['xlsx']}")

        print("\n🎉 Golden-path E2E run PASSED")
        return 0

    except AssertionError as exc:
        print(f"\n❌ ASSERTION FAILED: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        import traceback

        print(f"\n❌ UNEXPECTED ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
