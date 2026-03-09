# Deployment & Containerization Guide

Status: Active

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
- Optional deployment overrides:
	- `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_TRUSTED_HOST`, `PIP_NETWORK_CHECK_URL`, `PIP_NETWORK_TIMEOUT` for Docker builds that must use an internal PyPI mirror.
	- `SERVICE_IMAGE_NAME`, `SERVICE_IMAGE_TAG` to pin compose to a prebuilt image.
	- `SMOKE_USE_PREBUILT_IMAGE=1` when rerunning [scripts/e2e_smoke.sh](scripts/e2e_smoke.sh) against a pulled prebuilt image instead of building locally.
	- start from `.env.prebuilt.example` when the runtime must consume a CI-published image.
- Need a step-by-step dev loop? Refer to `docs/DOCKER_README.md` for hot reload expectations.

### 4.2 Prebuilt Image Path
- CI publishes the shared service image to `ghcr.io/<owner>/rag-eval` on `main` pushes.
- All service containers reuse that single image and switch entrypoints with the `SERVICE` environment variable.
- For the full operator flow, see [docs/prebuilt_image_workflow.md](docs/prebuilt_image_workflow.md).
- For an executable end-to-end runbook covering GHCR login, mirror env files, smoke commands, and success criteria, see [docs/runbooks/compose_e2e_operator_checklist.md](docs/runbooks/compose_e2e_operator_checklist.md).
- Convenience targets:
	- `make compose-prebuilt PREBUILT_ENV_FILE=.env.prebuilt`
	- `make smoke-prebuilt PREBUILT_ENV_FILE=.env.prebuilt`
	- start from `.env.mirror.example` when you need a dedicated mirror-backed compose env file.

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
- Multi-stage build: a `builder` stage creates an isolated venv; the final runtime stage only receives the virtualenv and service code, trimming unnecessary toolchain layers.
- Base image pinned to `python:3.11-slim-bookworm` and upgraded during build; security patches apply before the runtime is sealed.
- Runtime executes as the configurable non-root user (`rag`, UID/GID build args) with `/app`, `${MODELS_CACHE_PATH}`, and `${EXTENSIONS_DIR}` owned by that account.
- Dependency installation stays in a single layer with `pip --no-cache-dir`, `PIP_DISABLE_PIP_VERSION_CHECK`, and `PYTHONDONTWRITEBYTECODE` to avoid cache bloat and stray bytecode.
- Network-aware pip guard: build args (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_TRUSTED_HOST`, `PIP_NETWORK_CHECK_URL`, `PIP_NETWORK_TIMEOUT`) allow mirror selection and fail fast with `PyPI unreachable during image build` when no reachable package source exists, preventing broken partial images and long retry storms.
- Build arg `MODELS_CACHE` keeps the model cache path configurable (default `/var/cache/rag-models`) and is exposed as a named volume for persistence.
- Hardening verification steps and remediation checklist remain in `docs/hardening_checklist.md`.

## 8. Extensions & Plugin Loader (TASK-126)
- Host directory `extensions/` is mounted into `/extensions` within each container.
- The runtime loader (`services/common/plugin_loader.py`) enumerates `*.py` modules, expecting `PLUGIN_KIND` plus either a `register()` callable or `PLUGIN_DEFINITION` data payload.
- Sample metric plugin (`extensions/sample_metric.py`) demonstrates the contract—copy & adjust to introduce custom metrics or builders without rebuilding the base image.
- Override the mount location by exporting `EXTENSIONS_DIR=/custom/extensions` prior to running `make compose` or `make dev`.
- Validation: `pytest test_plugin_loader.py` exercises discovery logic and guards against regressions.

## 9. Remaining Roadmap
- TASK-127: Helm chart decomposition & toggleable services.
- TASK-128: Health/readiness probe standardisation across compose + Helm.

## 10. GPU Profile & Parity Validation
- GPU-enabled builds use `ENABLE_GPU=true` in the Docker build and Helm `gpu.*` values to switch the `processing` and `kg` services onto GPU-tagged images.
- Runtime sets `GPU_ENABLED=true` for those services and exposes `gpu_enabled{service=...}` in Prometheus output.
- `scripts/validate_dev_parity.py` now supports JSON/Markdown reports, snapshot comparison, extensions fingerprints, drift whitelisting, and separate Python drift severity for local vs CI usage.
- Recommended usage:
	- local workstation: `python3 scripts/validate_dev_parity.py --skip-installed-packages`
	- CI gate: `python3 scripts/validate_dev_parity.py --skip-installed-packages --python-drift-severity error`

## 11. Security Considerations
- Governance workflow now includes policy-as-code validation via OPA, gitleaks secret scanning, CycloneDX SBOM generation, SBOM diff output, and provenance emission.
- When `COSIGN_PRIVATE_KEY` is available in CI, images are signed automatically after push.
- Prebuilt image pull and signature verification examples are collected in [docs/prebuilt_image_workflow.md](docs/prebuilt_image_workflow.md).
