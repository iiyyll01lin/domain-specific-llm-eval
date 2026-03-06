#!/usr/bin/env python3
"""KG Threshold Tuning Harness (TASK-062d).

Evaluates relationship counts and average similarity scores across a grid of
thresholds for Jaccard, Overlap, and Cosine builders.  Outputs a JSON report
that guides configuration decisions.

Usage::

    python3 scripts/kg_threshold_tune.py --input path/to/nodes.json \\
        --output threshold_report.json

The input nodes.json must be an array of objects with at least the keys:

    node_id, entities, keyphrases

Optionally include "embedding" for cosine similarity tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from services.kg.relationships import (
    build_cosine_relationships,
    build_jaccard_relationships,
    build_overlap_relationships,
)


JACCARD_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
OVERLAP_THRESHOLDS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
COSINE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _avg_score(rels: List[Dict[str, Any]]) -> float:
    if not rels:
        return 0.0
    return sum(r["score"] for r in rels) / len(rels)


def sweep_jaccard(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for t in JACCARD_THRESHOLDS:
        rels = build_jaccard_relationships(nodes, threshold=t)
        results.append(
            {"threshold": t, "count": len(rels), "avg_score": round(_avg_score(rels), 4)}
        )
    return results


def sweep_overlap(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for t in OVERLAP_THRESHOLDS:
        rels = build_overlap_relationships(nodes, threshold=t)
        results.append(
            {"threshold": t, "count": len(rels), "avg_score": round(_avg_score(rels), 4)}
        )
    return results


def sweep_cosine(nodes: List[Dict[str, Any]], embedding_key: str = "embedding") -> List[Dict[str, Any]]:
    results = []
    for t in COSINE_THRESHOLDS:
        rels, skipped = build_cosine_relationships(nodes, threshold=t, embedding_key=embedding_key)
        results.append(
            {
                "threshold": t,
                "count": len(rels),
                "avg_score": round(_avg_score(rels), 4),
                "skipped_nodes": skipped,
            }
        )
    return results


def run_sweep(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "node_count": len(nodes),
        "jaccard": sweep_jaccard(nodes),
        "overlap": sweep_overlap(nodes),
        "cosine_embedding": sweep_cosine(nodes, "embedding"),
        "cosine_summary_embedding": sweep_cosine(nodes, "summary_embedding"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KG threshold tuning harness")
    parser.add_argument("--input", required=True, help="Path to nodes.json")
    parser.add_argument("--output", default="threshold_report.json", help="Output report path")
    args = parser.parse_args()

    nodes = json.loads(Path(args.input).read_text())
    if not isinstance(nodes, list):
        print("ERROR: nodes.json must be a JSON array", file=sys.stderr)
        sys.exit(1)

    report = run_sweep(nodes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {output_path}")
    print(f"  Nodes: {report['node_count']}")
    for builder in ("jaccard", "overlap", "cosine_embedding", "cosine_summary_embedding"):
        row = report[builder][0] if report[builder] else {}
        print(f"  {builder}: {len(report[builder])} threshold steps")


if __name__ == "__main__":
    main()
