#!/usr/bin/env python3
"""
Convert evaluation artifacts under an evaluations-pre directory into a portal-ready summary JSON.

Input: a folder like .../outputs/<run-id>/evaluations-pre containing
  - ragas_enhanced_detailed_calculations_*.json  (array of records)

Output: .../outputs/<run-id>/portal/ragas_enhanced_evaluation_results_<timestamp>_portal.json

Rules:
  - Group rows by question_index (fallback keys supported) into one item per question.
  - For each metric_name, average multiple scores if present.
  - Recognized metric names (case-sensitive):
      ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerSimilarity, ContextualKeywordMean
    Also supports snake_case variants by mapping.
  - Item shape:
      {
        "id": "<question_index>",
        "user_input": <question>,
        "reference": <ground_truth>,
        "rag_answer": <rag_answer>,
        "ContextPrecision": <number?>,
        ...metrics...
      }
  - Only stdlib; keep comments in English.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

KNOWN_METRICS = {
    "ContextPrecision": "ContextPrecision",
    "ContextRecall": "ContextRecall",
    "Faithfulness": "Faithfulness",
    "AnswerRelevancy": "AnswerRelevancy",
    "AnswerSimilarity": "AnswerSimilarity",
    "ContextualKeywordMean": "ContextualKeywordMean",
}

ALT_METRIC_KEYS = {
    "context_precision": "ContextPrecision",
    "context_recall": "ContextRecall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "answer_similarity": "AnswerSimilarity",
    "contextual_keyword_mean": "ContextualKeywordMean",
}

QUESTION_ID_CANDIDATES = ["question_index", "index", "row_id", "sample_id", "question_id", "id"]


def find_latest_detailed_file(eval_dir: Path) -> Path | None:
    pattern = re.compile(r"ragas_enhanced_detailed_calculations_.*\.json$")
    files = [p for p in eval_dir.iterdir() if p.is_file() and pattern.search(p.name)]
    if not files:
        return None
    # Pick the lexicographically latest file name, which usually correlates with timestamp suffix
    return sorted(files)[-1]


def normalize_metric_key(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    if name in KNOWN_METRICS:
        return KNOWN_METRICS[name]
    if name in ALT_METRIC_KEYS:
        return ALT_METRIC_KEYS[name]
    return None


def get_first(obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in obj:
            return obj[k]
    return None


def to_number(x: Any) -> float | None:
    try:
        if x is None:
            return None
        n = float(x)
        if n != n:  # NaN
            return None
        return n
    except Exception:
        return None


def aggregate_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Group by question index
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        qid = get_first(r, QUESTION_ID_CANDIDATES)
        if qid is None:
            # Skip if no question id can be found
            continue
        groups[str(qid)].append(r)

    items: List[Dict[str, Any]] = []
    for qid, rs in groups.items():
        # Prepare metric accumulators: metric -> list of scores
        acc: Dict[str, List[float]] = defaultdict(list)
        # Keep last seen texts as representative
        user_input = None
        reference = None
        rag_answer = None

        for r in rs:
            user_input = r.get("question", user_input)
            reference = r.get("ground_truth", reference)
            rag_answer = r.get("rag_answer", rag_answer)
            mk = normalize_metric_key(r.get("metric_name"))
            sc = to_number(r.get("score"))
            if mk is None or sc is None:
                continue
            acc[mk].append(sc)

        it: Dict[str, Any] = {
            "id": qid,
            "user_input": user_input,
            "reference": reference,
            "rag_answer": rag_answer,
        }
        # Average metric scores
        for mk, vals in acc.items():
            if not vals:
                continue
            it[mk] = sum(vals) / len(vals)

        items.append(it)

    # Sort by numeric id if possible
    def sort_key(d: Dict[str, Any]) -> Tuple[int, str]:
        try:
            return (int(d.get("id", 0)), str(d.get("id", "")))
        except Exception:
            return (0, str(d.get("id", "")))

    items.sort(key=sort_key)
    return items


def convert_eval_dir(eval_dir: Path, out_path: Path | None = None) -> Path:
    detailed = find_latest_detailed_file(eval_dir)
    if detailed is None:
        raise FileNotFoundError("No ragas_enhanced_detailed_calculations_*.json found in evaluations-pre")
    data = json.loads(detailed.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Detailed calculations JSON must be an array, got {type(data)} from {detailed}")
    items = aggregate_items(data)
    # Derive output path
    if out_path is None:
        run_dir = eval_dir.parent  # .../outputs/<run-id>
        portal_dir = run_dir / "portal"
        portal_dir.mkdir(parents=True, exist_ok=True)
        # Try to extract timestamp from filename
        m = re.search(r"(\d{8}_\d{6})", detailed.name)
        ts = m.group(1) if m else "unknown"
        out_path = portal_dir / f"ragas_enhanced_evaluation_results_{ts}_portal.json"
    out_content = {"items": items}
    out_path.write_text(json.dumps(out_content, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert evaluations-pre artifacts to portal-ready summary JSON")
    p.add_argument("eval_dir", type=str, help="Path to evaluations-pre directory")
    p.add_argument("--out", type=str, default=None, help="Output JSON path (optional)")
    args = p.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists() or not eval_dir.is_dir():
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")
    out_path = convert_eval_dir(eval_dir, Path(args.out) if args.out else None)
    print(f"Wrote portal summary: {out_path}")


if __name__ == "__main__":
    main()
