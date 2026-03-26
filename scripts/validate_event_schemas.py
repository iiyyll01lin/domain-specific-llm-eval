#!/usr/bin/env python3
"""Validate event schema registry integrity.

Checks:
 1. schema files exist
 2. sha256 matches (if not 'TBD') else populates suggestion
 3. no duplicate event name+version
 4. optional: basic JSON Schema structure keys
"""
from __future__ import annotations
import json, hashlib, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT/"events"/"schema_registry.json"

FAILURES = 0

def sha256(fp: pathlib.Path) -> str:
    h = hashlib.sha256()
    with fp.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def fail(msg: str):
    global FAILURES
    print(f"[FAIL] {msg}")
    FAILURES += 1

def main():
    if not REGISTRY.exists():
        print("Registry file missing")
        return 2
    data = json.loads(REGISTRY.read_text())
    events = data.get("events", [])
    seen = set()
    updated = False
    for ev in events:
        key = (ev["name"], ev["version"])
        if key in seen:
            fail(f"Duplicate event {key}")
        seen.add(key)
        schema_rel = ev.get("schema_file")
        if not schema_rel:
            fail(f"Missing schema_file for {ev}")
            continue
        schema_path = REGISTRY.parent / schema_rel
        if not schema_path.exists():
            fail(f"Schema file not found: {schema_rel}")
            continue
        digest = sha256(schema_path)
        if ev.get("sha256") in (None, "TBD"):
            ev["sha256"] = digest
            updated = True
        elif ev["sha256"] != digest:
            fail(f"Hash mismatch for {schema_rel}: registry={ev['sha256']} actual={digest}")
        # Light schema sanity
        try:
            schema_json = json.loads(schema_path.read_text())
        except Exception as e:
            fail(f"Invalid JSON in {schema_rel}: {e}")
            continue
        for req_key in ("type", "properties"):
            if req_key not in schema_json:
                fail(f"Schema missing '{req_key}' in {schema_rel}")
    if updated:
        REGISTRY.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print("Registry sha256 fields updated (TBD replaced). Commit the file.")
    if FAILURES:
        print(f"Validation failed with {FAILURES} error(s).")
        return 1
    print("All event schemas valid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
