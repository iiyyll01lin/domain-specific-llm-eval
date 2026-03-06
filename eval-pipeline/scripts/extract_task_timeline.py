#!/usr/bin/env python3
"""Extract task schedule (ID, Title, Status, Engineer, Target Sprint) from tasks docs.

Usage:
  python scripts/extract_task_timeline.py \
      --input docs/tasks/tasks.md \
      --output-format markdown csv \
      --out-dir reports

Outputs (examples):
  reports/task_timeline_<date>.md
  reports/task_timeline_<date>.csv

Design Notes:
  - Parses the markdown tables for ID/Title rows.
  - Then scans following fenced YAML governance blocks.
  - Governance block keys: status, engineer, target_sprint (others ignored here).
  - Gracefully handles missing fields (fills with '').
  - Can optionally merge Chinese file if provided (union by ID; prefer non-empty fields).
  - Stable ordering = appearance order in first file.
  - Suitable for periodic (cron) run; idempotent file naming via timestamp unless --no-timestamp.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import json
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

TASK_ROW_RE = re.compile(r"^\|\s*(TASK-\d+[a-d]?)\s*\|\s*([^|]+?)\s*\|", re.IGNORECASE)
GOVERNANCE_HEADER_RE = re.compile(r"^`{3}yaml\s*$")
TASK_GOVERNANCE_ID_RE = re.compile(r"^#\s*(TASK-\d+[a-d]?)\s+Governance", re.IGNORECASE)
KEY_VALUE_RE = re.compile(r"^\s*([a-zA-Z0-9_]+):\s*(.*)$")


@dataclass
class TaskRecord:
    task_id: str
    title: str
    status: str = ""
    engineer: str = ""
    target_sprint: str = ""

    def merge(self, other: "TaskRecord") -> None:
        # Prefer existing non-empty; fill missing from other
        for f in ("title", "status", "engineer", "target_sprint"):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))


def parse_tasks(md_text: str) -> Dict[str, TaskRecord]:
    lines = md_text.splitlines()
    tasks: Dict[str, TaskRecord] = {}
    order: List[str] = []

    # Phase 1: capture table task rows
    for line in lines:
        m = TASK_ROW_RE.match(line)
        if m:
            task_id, title = m.groups()
            title = title.strip()
            if task_id not in tasks:
                tasks[task_id] = TaskRecord(task_id=task_id, title=title)
                order.append(task_id)

    # Phase 2: governance blocks
    current_id: Optional[str] = None
    in_yaml = False
    for i, line in enumerate(lines):
        # Identify governance header comments preceding yaml block
        header_match = TASK_GOVERNANCE_ID_RE.match(line.strip())
        if header_match:
            current_id = header_match.group(1)
            continue

        if GOVERNANCE_HEADER_RE.match(line):
            in_yaml = True
            continue
        if in_yaml and line.strip().startswith("```"):
            in_yaml = False
            current_id = None
            continue
        if in_yaml and current_id and current_id in tasks:
            kv = KEY_VALUE_RE.match(line)
            if kv:
                k, v = kv.groups()
                k_l = k.lower()
                v = v.strip().strip('"')
                if k_l in ("status", "engineer", "target_sprint"):
                    # Remove inline comments after '#'
                    v_clean = v.split("#", 1)[0].strip()
                    setattr(tasks[current_id], k_l, v_clean)

    # Preserve original order
    return {tid: tasks[tid] for tid in order}


def merge_task_sets(
    base: Dict[str, TaskRecord], other: Dict[str, TaskRecord]
) -> Dict[str, TaskRecord]:
    for tid, rec in other.items():
        if tid in base:
            base[tid].merge(rec)
        else:
            base[tid] = rec
    return base


def _summarize(tasks: Dict[str, TaskRecord]) -> str:
    total = len(tasks)
    sub_task_pattern = re.compile(r"TASK-\d+[a-d]$")
    main_tasks = sum(1 for tid in tasks if not sub_task_pattern.match(tid))
    sub_tasks = total - main_tasks
    sprint_counter: Counter[str] = Counter()
    engineer_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for rec in tasks.values():
        if rec.target_sprint:
            sprint_counter[rec.target_sprint] += 1
        if rec.engineer:
            engineer_counter[rec.engineer] += 1
        if rec.status:
            status_counter[rec.status] += 1
    lines: List[str] = []
    lines.append("## Tasks Summary")
    lines.append("")
    lines.append(f"Total Tasks: {total}")
    lines.append(f"Main Tasks: {main_tasks}")
    lines.append(f"Sub Tasks: {sub_tasks}")
    if sprint_counter:
        lines.append("")
        lines.append("### Tasks per Sprint")
        for s in sorted(
            sprint_counter, key=lambda x: (int(x) if x.isdigit() else 999, x)
        ):
            lines.append(f"- Sprint {s}: {sprint_counter[s]}")
    if engineer_counter:
        lines.append("")
        lines.append("### Tasks per Engineer")
        for eng in sorted(engineer_counter):
            lines.append(f"- {eng}: {engineer_counter[eng]}")
    if status_counter:
        lines.append("")
        lines.append("### Tasks per Status")
        for st in sorted(status_counter):
            lines.append(f"- {st}: {status_counter[st]}")
    return "\n".join(lines) + "\n"


def write_markdown(tasks: Dict[str, TaskRecord], path: Path) -> None:
    headers = ["ID", "Title", "Status", "Engineer", "Target Sprint"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for rec in tasks.values():
        lines.append(
            f"| {rec.task_id} | {rec.title} | {rec.status} | {rec.engineer} | {rec.target_sprint} |"
        )
    lines.append("")
    lines.append(_summarize(tasks))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(tasks: Dict[str, TaskRecord], path: Path) -> None:
    fieldnames = ["task_id", "title", "status", "engineer", "target_sprint"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in tasks.values():
            writer.writerow(asdict(rec))


def write_json(tasks: Dict[str, TaskRecord], path: Path) -> None:
    payload = [asdict(rec) for rec in tasks.values()]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _inherit_subtasks(tasks: Dict[str, TaskRecord]) -> None:
    """Fill empty engineer/target_sprint for subtasks (TASK-###a-d) from their parent TASK-###.

    Only inherit if subtask field empty and parent has value.
    """
    sub_re = re.compile(r"^(TASK-\d+)([a-d])$")
    for tid, rec in tasks.items():
        m = sub_re.match(tid)
        if not m:
            continue
        parent_id = m.group(1)
        parent = tasks.get(parent_id)
        if not parent:
            continue
        if not rec.engineer and parent.engineer:
            rec.engineer = parent.engineer
        if not rec.target_sprint and parent.target_sprint:
            rec.target_sprint = parent.target_sprint
        if not rec.status and parent.status:
            rec.status = parent.status


def main():
    ap = argparse.ArgumentParser(description="Extract task timeline scheduling report")
    ap.add_argument(
        "--input",
        required=True,
        help="Primary tasks markdown file (e.g., docs/tasks/tasks.md)",
    )
    ap.add_argument("--input-zh", help="Optional Chinese tasks markdown to merge")
    ap.add_argument(
        "--out-dir", default="reports", help="Output directory (default: reports)"
    )
    ap.add_argument(
        "--output-format",
        nargs="+",
        default=["markdown"],
        choices=["markdown", "csv", "json"],
        help="Formats to output",
    )
    ap.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't append timestamp to filenames",
    )
    ap.add_argument(
        "--overrides",
        help="YAML file mapping task_id -> {status, engineer, target_sprint}",
    )
    ap.add_argument(
        "--inherit-subtasks",
        action="store_true",
        help="Inherit engineer/target_sprint/status for TASK-###a-d from parent TASK-### if missing",
    )
    ap.add_argument(
        "--sprint-summary-json",
        action="store_true",
        help="Emit sprint_summary.json aggregation for burndown data",
    )
    args = ap.parse_args()

    base_path = Path(args.input)
    if not base_path.exists():
        raise SystemExit(f"Input file not found: {base_path}")
    primary_text = base_path.read_text(encoding="utf-8")
    tasks = parse_tasks(primary_text)

    if args.input_zh:
        zh_path = Path(args.input_zh)
        if zh_path.exists():
            zh_text = zh_path.read_text(encoding="utf-8")
            tasks = merge_task_sets(tasks, parse_tasks(zh_text))

    if args.overrides:
        if not yaml:
            raise SystemExit("pyyaml not installed; install or omit --overrides")
        ov_path = Path(args.overrides)
        if ov_path.exists():
            data = yaml.safe_load(ov_path.read_text(encoding="utf-8")) or {}
            for tid, fields in data.items():
                if tid not in tasks:
                    # Create stub if override references future task
                    tasks[tid] = TaskRecord(task_id=tid, title=fields.get("title", tid))
                rec = tasks[tid]
                for k in ("status", "engineer", "target_sprint"):
                    val = fields.get(k)
                    if val:
                        setattr(rec, k, val)

    if args.inherit_subtasks:
        _inherit_subtasks(tasks)

    def build_sprint_summary(tasks: Dict[str, TaskRecord]) -> Dict[str, object]:
        sprint_map: Dict[str, List[TaskRecord]] = {}
        # Pattern for main tasks (no subletter a-d)
        main_pattern = re.compile(r"^TASK-\d+$")
        done_set = {"Done", "Verified"}
        for rec in tasks.values():
            if not rec.target_sprint:
                continue
            sprint_map.setdefault(rec.target_sprint, []).append(rec)

        def sprint_sort_key(x: str) -> Tuple[int, str]:
            return (int(x), x) if x.isdigit() else (9999, x)

        cumulative_done_main = 0
        cumulative_done_all = 0
        sprint_entries: List[Dict[str, object]] = []
        for sprint in sorted(sprint_map, key=sprint_sort_key):
            bucket = sprint_map[sprint]
            total = len(bucket)
            main = [r for r in bucket if main_pattern.match(r.task_id)]
            sub_count = total - len(main)
            by_status: Counter[str] = Counter(r.status or "" for r in bucket)
            done_main = sum(1 for r in main if r.status in done_set)
            done_all = sum(1 for r in bucket if r.status in done_set)
            remaining_main = len(main) - done_main
            remaining_all = total - done_all
            cumulative_done_main += done_main
            cumulative_done_all += done_all
            eng_counter: Counter[str] = Counter(
                r.engineer for r in bucket if r.engineer
            )
            sprint_entries.append(
                {
                    "sprint": sprint,
                    "total_tasks": total,
                    "main_tasks": len(main),
                    "sub_tasks": sub_count,
                    "by_status": dict(by_status),
                    "done_main": done_main,
                    "done_all": done_all,
                    "remaining_main": remaining_main,
                    "remaining_all": remaining_all,
                    "cumulative_done_main": cumulative_done_main,
                    "cumulative_done_all": cumulative_done_all,
                    "engineers": dict(eng_counter),
                    "velocity_candidate": done_main,
                    "completion_ratio": (done_all / total) if total else 0.0,
                }
            )
        totals_main = sum(1 for r in tasks.values() if main_pattern.match(r.task_id))
        totals_all = len(tasks)
        return {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "sprints": sprint_entries,
            "totals": {
                "all_main": totals_main,
                "all_sub": totals_all - totals_main,
                "all_tasks": totals_all,
            },
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        "" if args.no_timestamp else dt.datetime.utcnow().strftime("_%Y%m%d_%H%M%S")
    )

    if "markdown" in args.output_format:
        write_markdown(tasks, out_dir / f"task_timeline{timestamp}.md")
    if "csv" in args.output_format:
        write_csv(tasks, out_dir / f"task_timeline{timestamp}.csv")
    if "json" in args.output_format:
        write_json(tasks, out_dir / f"task_timeline{timestamp}.json")
    if args.sprint_summary_json:
        summary = build_sprint_summary(tasks)
        (out_dir / f"sprint_summary{timestamp}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Extracted {len(tasks)} tasks -> {out_dir}")


if __name__ == "__main__":
    main()
