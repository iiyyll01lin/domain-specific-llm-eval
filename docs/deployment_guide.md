# Deployment & Containerization Guide

Status: Draft

## 1. Overview
Multi-service architecture packaged via a single base image parameterized by `SERVICE` env. Compose and Helm (future) orchestrate runtime wiring. This guide covers local dev, tagging, and environment parity.

## 2. Services
| Service    | Purpose (skeleton)                            |
|------------|-----------------------------------------------|
| ingestion  | Accept document refs (future POST /documents) |
| processing | Chunking & embeddings pipeline (future)       |
| testset    | Testset generation jobs                       |
| eval       | Evaluation runner                             |
| reporting  | Report artifact generation                    |
| adapter    | Insights normalization                        |
| kg         | Knowledge graph builder (flagged)             |

## 3. Build
```
make build               # builds base image as rag-eval:dev
make build-tag           # builds (if needed) then tags vX.Y.Z & git-<sha>
DRY_RUN=1 make tag       # preview the tags without mutating local images
```
Version comes from `VERSION` file; override with `VERSION=0.2.0 make build-tag`.

## 4. Compose Profiles
Baseline (no hot reload):
```
make compose
```
Dev (hot reload & bind mount):
```
make dev
```
Hot reload override uses `docker-compose.dev.override.yml` layering in volumes & `--reload` commands.

### 4.1 Environment Files
- Default runtime variables load from `.env.compose`. The compose file accepts an override by exporting `COMPOSE_ENV_FILE` (e.g., `export COMPOSE_ENV_FILE=.env.local`).
- Keep secrets out of version control—create an external file and point the environment variable to it before running `make compose`/`make dev`.
- Common defaults provided:
	- `PYTHONPATH=/app`
	- `LOG_LEVEL=INFO`
	- `SERVICE_HOST=0.0.0.0`
	- `SERVICE_PORT=8000`
- Need a step-by-step dev loop? Refer to `docs/DOCKER_README.md` for hot reload expectations.

## 5. Tagging Strategy
- Semantic tag: `vX.Y.Z` (from VERSION file)
- Source tag: `git-<short_sha>`
- Dev tag: `dev` (ephemeral)

Example:
```
make build-tag IMAGE_NAME=rag-eval VERSION=0.1.1
```

## 6. Validation
Compose service coverage check:
```
make validate-compose
```
Outputs JSON with failure reasons (non-zero exit on problems).

## 7. Image Hardening Summary (TASK-125)
- Uvicorn now runs as non-root user (`rag`, UID/GID configurable via build args).
- Dependency install layers consolidated with `pip --no-cache-dir` enforced.
- Configurable model cache path (`MODELS_CACHE` build arg) with dedicated volume mount.
- Default PATH extends `${HOME}/.local/bin` so user-level installs resolve.
- Ownership of `/app` and cache directory transferred to the service user at build time.
- Checklist with verification steps lives in `docs/hardening_checklist.md`.

## 8. Extensions & Plugin Loader (TASK-126)
- Host directory `extensions/` is mounted into `/extensions` within each container.
- The runtime loader (`services/common/plugin_loader.py`) enumerates `*.py` modules, expecting `PLUGIN_KIND` plus either a `register()` callable or `PLUGIN_DEFINITION` data payload.
- Sample metric plugin (`extensions/sample_metric.py`) demonstrates the contract—copy & adjust to introduce custom metrics or builders without rebuilding the base image.
- Override the mount location by exporting `EXTENSIONS_DIR=/custom/extensions` prior to running `make compose` or `make dev`.
- Validation: `pytest test_plugin_loader.py` exercises discovery logic and guards against regressions.

## 9. Remaining Roadmap
- TASK-127: Helm chart decomposition & toggleable services.
- TASK-128: Health/readiness probe standardisation across compose + Helm.

## 10. Security Considerations (Preview)
Upcoming governance work will add SBOM & Trivy integration, optional Cosign signing, and policy-as-code gates.
