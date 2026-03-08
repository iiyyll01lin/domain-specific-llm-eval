#!/usr/bin/env bash
# TASK-101: E2E Pipeline Smoke Test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/e2e_smoke.py
