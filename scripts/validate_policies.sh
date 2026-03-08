#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v opa >/dev/null 2>&1; then
  echo "OPA is required for policy validation. Install it or run via CI." >&2
  exit 1
fi

cd "$ROOT_DIR"

opa test policy

python3 <<'PY' > /tmp/policy_inputs.json
import json
from pathlib import Path

taxonomy = json.loads(Path("telemetry/telemetry_taxonomy.json").read_text(encoding="utf-8"))
payloads = {
    "events": [{"event_key": item["key"]} for item in taxonomy.get("events", [])],
    "metrics": [{"metric_name": item["name"]} for item in taxonomy.get("metrics", [])],
}
print(json.dumps(payloads))
PY

python3 <<'PY'
import json
import subprocess
import sys
from pathlib import Path

payloads = json.loads(Path("/tmp/policy_inputs.json").read_text(encoding="utf-8"))

for event in payloads["events"]:
    result = subprocess.run(
        ["opa", "eval", "-f", "json", "-d", "policy", "-i", "/dev/stdin", "data.naming.deny"],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    if data["result"][0]["expressions"][0]["value"]:
        print(f"Naming policy rejected event payload: {event}", file=sys.stderr)
        sys.exit(1)

for metric in payloads["metrics"]:
    result = subprocess.run(
        ["opa", "eval", "-f", "json", "-d", "policy", "-i", "/dev/stdin", "data.naming.deny"],
        input=json.dumps(metric),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    if data["result"][0]["expressions"][0]["value"]:
        print(f"Naming policy rejected metric payload: {metric}", file=sys.stderr)
        sys.exit(1)

result = subprocess.run(
    ["opa", "eval", "-f", "json", "-d", "policy", "-i", "events/schema_registry.json", "data.schema_registry.deny"],
    capture_output=True,
    text=True,
    check=True,
)
data = json.loads(result.stdout)
if data["result"][0]["expressions"][0]["value"]:
    print("Schema registry policy rejected current registry", file=sys.stderr)
    sys.exit(1)

print("OPA governance policy validation passed")
PY