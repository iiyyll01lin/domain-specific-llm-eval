#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_POLICY_INPUTS="$(mktemp)"
export TMP_POLICY_INPUTS
OPA_CMD=(opa)

cleanup() {
    rm -f "$TMP_POLICY_INPUTS"
}

trap cleanup EXIT

if ! command -v opa >/dev/null 2>&1; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "OPA is not installed and docker is unavailable for fallback execution." >&2
        exit 1
    fi
    echo "OPA binary not found; using dockerized OPA image via docker." >&2
    OPA_CMD=(docker run --rm -i -v "$ROOT_DIR:/workspace" -w /workspace openpolicyagent/opa:latest)
fi

OPA_CMD_JSON="$(printf '%s\n' "${OPA_CMD[@]}" | python3 -c 'import json, sys; print(json.dumps([line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]))')"
export OPA_CMD_JSON

cd "$ROOT_DIR"

"${OPA_CMD[@]}" test policy

python3 <<'PY' > "$TMP_POLICY_INPUTS"
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
import os
import subprocess
import sys
from pathlib import Path

payloads = json.loads(Path(os.environ["TMP_POLICY_INPUTS"]).read_text(encoding="utf-8"))
opa_cmd = json.loads(os.environ["OPA_CMD_JSON"])

for event in payloads["events"]:
    result = subprocess.run(
        [*opa_cmd, "eval", "-f", "json", "-d", "policy", "-i", "/dev/stdin", "data.naming.deny"],
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
        [*opa_cmd, "eval", "-f", "json", "-d", "policy", "-i", "/dev/stdin", "data.naming.deny"],
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
    [*opa_cmd, "eval", "-f", "json", "-d", "policy", "-i", "events/schema_registry.json", "data.schema_registry.deny"],
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