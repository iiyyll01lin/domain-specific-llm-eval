#!/usr/bin/env bash
set -euo pipefail
# Build and run the PDF service container locally.
# Comments in English only.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
IMG=insights-pdf-service:dev

docker build -t "$IMG" .
docker run --rm -p 8787:8787 "$IMG"
