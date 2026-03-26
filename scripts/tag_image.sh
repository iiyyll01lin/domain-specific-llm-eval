#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rag-eval}"
VERSION_FILE="${VERSION_FILE:-VERSION}"
if [ -f "$VERSION_FILE" ]; then
  VERSION=$(cat "$VERSION_FILE")
else
  VERSION="${VERSION:-0.1.0}"
fi
GIT_SHA=$(git rev-parse --short HEAD)
BASE_TAG="${BASE_TAG:-dev}"
DRY_RUN="${DRY_RUN:-0}"

echo "Tagging image ${IMAGE_NAME}:${BASE_TAG} as:"
echo " - ${IMAGE_NAME}:v${VERSION}"
echo " - ${IMAGE_NAME}:git-${GIT_SHA}"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY-RUN] docker tag ${IMAGE_NAME}:${BASE_TAG} ${IMAGE_NAME}:v${VERSION}"
  echo "[DRY-RUN] docker tag ${IMAGE_NAME}:${BASE_TAG} ${IMAGE_NAME}:git-${GIT_SHA}"
else
  docker tag "${IMAGE_NAME}:${BASE_TAG}" "${IMAGE_NAME}:v${VERSION}"
  docker tag "${IMAGE_NAME}:${BASE_TAG}" "${IMAGE_NAME}:git-${GIT_SHA}"
fi

echo "Done."
