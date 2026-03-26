#!/usr/bin/env python3
"""Validate telemetry taxonomy for naming and duplicates.
"""
from __future__ import annotations
import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TAX = ROOT/"telemetry"/"telemetry_taxonomy.json"

FAILURES = 0

def fail(msg: str):
    global FAILURES
    print(f"[FAIL] {msg}")
    FAILURES += 1

def main():
    if not TAX.exists():
        print("telemetry_taxonomy.json missing")
        return 2
    data = json.loads(TAX.read_text())
    rules = data.get("validation_rules", {})
    ev_re = re.compile(rules.get("event_key_regex", r"^.+$"))
    met_re = re.compile(rules.get("metric_name_regex", r"^.+$"))
    events = data.get("events", [])
    metrics = data.get("metrics", [])

    ev_keys = [e["key"] for e in events]
    if len(ev_keys) != len(set(ev_keys)) and rules.get("no_duplicate_events", True):
        fail("Duplicate event keys detected")
    for k in ev_keys:
        if not ev_re.match(k):
            fail(f"Event key regex mismatch: {k}")
    metric_names = [m["name"] for m in metrics]
    if len(metric_names) != len(set(metric_names)) and rules.get("no_duplicate_metrics", True):
        fail("Duplicate metric names detected")
    for k in metric_names:
        if not met_re.match(k):
            fail(f"Metric name regex mismatch: {k}")
    if FAILURES:
        print(f"Telemetry taxonomy validation failed with {FAILURES} error(s).")
        return 1
    print("Telemetry taxonomy valid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
