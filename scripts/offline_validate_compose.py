#!/usr/bin/env python3
"""Offline validator for docker-compose.services.yml without invoking docker.
Checks:
  - Required services present
  - Each service has environment SERVICE set (in runtime our image uses it via command or env)
  - Healthcheck block present
Outputs JSON summary and exits non-zero on failure.
"""
import sys, json, re
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # lightweight fallback if PyYAML not installed
    print(json.dumps({"error": "Missing PyYAML dependency"}))
    sys.exit(1)

COMPOSE_FILE = Path("docker-compose.services.yml")
REQUIRED = ["ingestion","processing","testset","eval","reporting","adapter","kg"]

def load():
    if not COMPOSE_FILE.exists():
        raise SystemExit(f"compose file not found: {COMPOSE_FILE}")
    with COMPOSE_FILE.open() as f:
        return yaml.safe_load(f)

def main():
    data = load()
    services = data.get("services", {})
    failures = []
    for name in REQUIRED:
        if name not in services:
            failures.append(f"missing service: {name}")
            continue
        svc = services[name]
        env = svc.get("environment", {}) or {}
        if "SERVICE" not in env:
            # Not strictly required if command sets module path, but we expect it here for clarity
            failures.append(f"service {name} missing SERVICE env")
        if "healthcheck" not in svc:
            failures.append(f"service {name} missing healthcheck")
    result = {"required": REQUIRED, "present": list(services.keys()), "failures": failures}
    print(json.dumps(result, indent=2))
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
