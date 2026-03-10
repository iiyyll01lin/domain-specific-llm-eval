#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DOCS = {
    "en": ROOT / "eval-pipeline" / "docs" / "tasks" / "tasks.md",
    "zh": ROOT / "eval-pipeline" / "docs" / "tasks" / "tasks.zh.md",
}

GOVERNANCE_HEADER_PATTERN = re.compile(
    r"^# (?P<task>TASK-\d+[a-z]?)\s+(?:Governance|治理)\s*$", re.MULTILINE
)
STATUS_PATTERN = re.compile(r"^\s*status:\s*(?P<status>[A-Za-z-]+)(?:\s+#.*)?\s*$", re.MULTILINE)
CODE_FENCE_PATTERN = re.compile(r"^```(?:yaml)?\s*$", re.MULTILINE)
YAML_BLOCK_PATTERN = re.compile(r"```yaml\n(?P<body>.*?)\n```", re.DOTALL)


def parse_governance_sections(text: str):
    sections = {}
    duplicates = []

    for yaml_block in YAML_BLOCK_PATTERN.finditer(text):
        block_text = yaml_block.group("body")
        matches = list(GOVERNANCE_HEADER_PATTERN.finditer(block_text))
        for index, match in enumerate(matches):
            task = match.group("task")
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(block_text)
            body = block_text[start:end]
            if task in sections:
                duplicates.append(task)
                continue
            sections[task] = {
                "body": body,
                "statuses": STATUS_PATTERN.findall(body),
            }

    return sections, duplicates


def validate_doc(doc_path: Path):
    failures = []
    text = doc_path.read_text(encoding="utf-8")
    fences = CODE_FENCE_PATTERN.findall(text)
    if len(fences) % 2 != 0:
        failures.append(f"{doc_path.name}: unmatched markdown code fence count ({len(fences)})")

    sections, duplicates = parse_governance_sections(text)
    duplicate_counts = Counter(duplicates)
    for task, count in sorted(duplicate_counts.items()):
        failures.append(f"{doc_path.name}: duplicate governance header for {task} ({count + 1} occurrences)")

    for task, details in sorted(sections.items()):
        statuses = details["statuses"]
        if not statuses:
            failures.append(f"{doc_path.name}: {task} missing status line")
        elif len(set(statuses)) != 1:
            failures.append(f"{doc_path.name}: {task} has inconsistent statuses {sorted(set(statuses))}")

    return sections, failures


def main() -> int:
    failures = []
    parsed = {}

    for locale, doc_path in TASK_DOCS.items():
        sections, doc_failures = validate_doc(doc_path)
        parsed[locale] = sections
        failures.extend(doc_failures)

    en_tasks = set(parsed["en"])
    zh_tasks = set(parsed["zh"])
    for task in sorted(en_tasks - zh_tasks):
        failures.append(f"tasks.zh.md: missing governance section for {task}")
    for task in sorted(zh_tasks - en_tasks):
        failures.append(f"tasks.md: missing governance section for {task}")

    for task in sorted(en_tasks & zh_tasks):
        en_statuses = parsed["en"][task]["statuses"]
        zh_statuses = parsed["zh"][task]["statuses"]
        if not en_statuses or not zh_statuses:
            continue
        en_status = en_statuses[0]
        zh_status = zh_statuses[0]
        if en_status != zh_status:
            failures.append(f"status mismatch for {task}: tasks.md={en_status}, tasks.zh.md={zh_status}")

    if failures:
        print("Task governance validation failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Task governance validation passed.")
    print(f" - governance sections: {len(parsed['en'])} matched task IDs")
    print(" - full english/chinese status alignment: OK")
    print(" - markdown governance structure: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
