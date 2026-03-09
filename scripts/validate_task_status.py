#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DOCS = [
    ROOT / "eval-pipeline" / "docs" / "tasks" / "tasks.md",
    ROOT / "eval-pipeline" / "docs" / "tasks" / "tasks.zh.md",
]
EXPECTED = {
    "TASK-120": "Verified",
    "TASK-121": "Verified",
    "TASK-122": "Verified",
    "TASK-125": "Verified",
}

section_pattern = re.compile(
    r"^# (?P<task>TASK-\d+)[^\n]*\n(?P<body>.*?)(?=^# TASK-|\Z)", re.MULTILINE | re.DOTALL
)
status_pattern = re.compile(r"status:\s*(?P<status>\w+)")

failures: list[str] = []

for doc in TASK_DOCS:
    text = doc.read_text(encoding="utf-8")
    sections = {match.group("task"): match.group("body") for match in section_pattern.finditer(text)}

    for task, expected_status in EXPECTED.items():
        body = sections.get(task)
        if body is None:
            failures.append(f"{doc.name}: missing section for {task}")
            continue
        statuses = status_pattern.findall(body)
        if not statuses:
            failures.append(f"{doc.name}: {task} missing status line")
            continue
        invalid = [status for status in statuses if status != expected_status]
        if invalid:
            failures.append(
                f"{doc.name}: {task} expected {expected_status}, found {set(invalid)}"
            )

if failures:
    print("Task governance validation failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("Task governance validation passed for:")
for task, status in EXPECTED.items():
    print(f" - {task} -> {status}")
