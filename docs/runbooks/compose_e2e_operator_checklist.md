# Compose E2E Operator Checklist

Status: Active

## Purpose
Use this checklist to make the compose-backed service stack runnable in environments where direct Docker builds may fail because PyPI is unreachable, or where service images must be pulled from GHCR.

## A. Preflight
- [ ] Docker daemon is running.
- [ ] `docker compose version` works.
- [ ] Python 3 is available for local validators.
- [ ] If `insights-portal/` will be built locally, Node.js 18.18+ or 20.x LTS is active.
- [ ] You know which path you will use:
  - [ ] Build locally with a PyPI mirror
  - [ ] Reuse a prebuilt GHCR image

## B. GHCR Prebuilt Image Path
### B1. Authentication
- [ ] Confirm the package owner and image name: `ghcr.io/<owner>/rag-eval`
- [ ] If the package is private, create a GitHub PAT with `read:packages`.
- [ ] Log in:

```bash
export CR_PAT=<your_pat>
printf '%s' "$CR_PAT" | docker login ghcr.io -u <github-username> --password-stdin
```

- [ ] If your organization uses SSO, ensure the PAT is SSO-authorized.

### B2. Tag Selection
The workflow publishes these tags on `main` pushes:
- `dev`
- `v<VERSION>`
- `git-<short_sha>`

Verify a tag is readable:

```bash
docker manifest inspect ghcr.io/<owner>/rag-eval:dev
```

Success condition:
- [ ] `docker manifest inspect` returns JSON instead of `denied`

### B3. Compose Env File
- [ ] Copy the template:

```bash
cp .env.prebuilt.example .env.prebuilt
```

- [ ] Fill in:
  - [ ] `SERVICE_IMAGE_NAME=ghcr.io/<owner>/rag-eval`
  - [ ] `SERVICE_IMAGE_TAG=<chosen-tag>`
  - [ ] Object store values if they differ from local MinIO defaults

### B4. Pull + Smoke
- [ ] Pull once:

```bash
COMPOSE_ENV_FILE=.env.prebuilt docker compose --env-file .env.prebuilt -f docker-compose.services.yml pull ingestion processing testset eval reporting
```

- [ ] Run the compose-backed smoke:

```bash
COMPOSE_ENV_FILE=.env.prebuilt SMOKE_USE_PREBUILT_IMAGE=1 bash scripts/e2e_smoke.sh
```

Success condition:
- [ ] All services become healthy
- [ ] Smoke prints `E2E smoke test passed`

## C. Local Build With PyPI Mirror Path
### C1. Mirror Inputs
You need your organization-specific values for:
- `PIP_INDEX_URL`
- `PIP_EXTRA_INDEX_URL` (optional)
- `PIP_TRUSTED_HOST` (optional, often required for HTTP or custom TLS)
- `PIP_NETWORK_CHECK_URL` (optional; usually point this at the same mirror)
- `PIP_NETWORK_TIMEOUT` (optional)

### C2. Create Local Override File
- [ ] Create a private env file such as `.env.local` or `.env.mirror`.
- [ ] Add values like:

```dotenv
PIP_INDEX_URL=https://<internal-mirror>/simple
PIP_EXTRA_INDEX_URL=
PIP_TRUSTED_HOST=<internal-mirror-host>
PIP_NETWORK_CHECK_URL=https://<internal-mirror>/simple/
PIP_NETWORK_TIMEOUT=10
```

### C3. Preflight Mirror Reachability
- [ ] Confirm the mirror responds from the host:

```bash
curl -I https://<internal-mirror>/simple/
```

Success condition:
- [ ] Returns HTTP 200/301/302 and does not time out

### C4. Build + Smoke
- [ ] Run compose with the mirror env file:

```bash
COMPOSE_ENV_FILE=.env.mirror docker compose --env-file .env.mirror -f docker-compose.services.yml build
COMPOSE_ENV_FILE=.env.mirror bash scripts/e2e_smoke.sh
```

Success condition:
- [ ] Docker build no longer prints `PyPI unreachable during image build`
- [ ] Smoke prints `E2E smoke test passed`

## D. Governance / Regression Validation
Run after either path succeeds:

```bash
pytest services/tests -q
python3 scripts/validate_event_schemas.py
python3 scripts/validate_telemetry_taxonomy.py
python3 scripts/validate_task_status.py
python3 scripts/validate_dev_parity.py --skip-installed-packages
bash scripts/validate_policies.sh
make validate-compose
python3 scripts/e2e_smoke.py
```

Success condition:
- [ ] All commands exit 0
- [ ] Local parity may still show a Python drift warning when host Python is 3.10 and the image target is 3.11

## E. Frontend Readiness
- [ ] Use Node.js 18.18+ or 20.x LTS
- [ ] Install dependencies:

```bash
export PATH="$HOME/.local/node-v20/bin:$PATH"
cd insights-portal
npm install
```

- [ ] Run tests/build:

```bash
npm run test -- --run
npm run build
```

Success condition:
- [ ] Vitest passes
- [ ] Vite build completes

## F. Troubleshooting
### GHCR returns `denied`
- [ ] Re-run `docker login ghcr.io`
- [ ] Verify the PAT has `read:packages`
- [ ] Verify package visibility and owner
- [ ] Verify the selected tag exists

### Compose build still fails on PyPI
- [ ] Confirm `PIP_INDEX_URL` is actually present in the env file used by compose
- [ ] Confirm `PIP_NETWORK_CHECK_URL` points to the same reachable mirror
- [ ] Confirm the mirror serves all required wheels/sdists from `requirements.txt`

### Frontend build fails immediately with syntax errors inside TypeScript/Vite tooling
- [ ] Check `node -v`
- [ ] If it is older than 18.18, switch to the local Node 20 runtime first
