# Prebuilt Service Image Workflow

Status: Active

## 1. Purpose
Use this workflow when the current environment cannot build the service image locally because Docker cannot reach PyPI. In this repository, all service containers share one image and switch behavior through the `SERVICE` environment variable in [docker-compose.services.yml](docker-compose.services.yml).

The CI publication target is:
- Registry: `ghcr.io`
- Image: `ghcr.io/<owner>/rag-eval`
- Tags:
  - `dev`
  - `v<VERSION>`
  - `git-<short_sha>`

## 2. When CI Publishes Images
The publish path is defined in [.github/workflows/build-governance.yml](.github/workflows/build-governance.yml):
- validators run first;
- image build, SBOM, scan, and optional signing run next;
- images are pushed only on `main` branch pushes.

If you need a fresh prebuilt image, merge or cherry-pick the desired commit into a connected CI environment that can run that workflow.

## 3. Authentication
If the GHCR package is private, log in first:

```bash
docker login ghcr.io -u <github-username>
```

Use a GitHub PAT or other credential with package read access.

## 4. Local Environment File
Start from the example:

```bash
cp .env.prebuilt.example .env.prebuilt
$EDITOR .env.prebuilt
```

Required values:
- `SERVICE_IMAGE_NAME=ghcr.io/<owner>/rag-eval`
- `SERVICE_IMAGE_TAG=v<VERSION>` or `git-<short_sha>`
- object store settings matching your environment

Keep `COMPOSE_ENV_FILE=.env.prebuilt` for all commands below.

## 5. Pull and Verify the Image
Pull the shared runtime image:

```bash
docker pull ghcr.io/<owner>/rag-eval:v<VERSION>
```

Optional signature verification if `cosign.pub` is available:

```bash
cosign verify --key cosign.pub ghcr.io/<owner>/rag-eval:v<VERSION>
```

Optional inspection:

```bash
docker image inspect ghcr.io/<owner>/rag-eval:v<VERSION>
```

## 6. Start the Compose Stack Without Building
Use the prebuilt image path:

```bash
COMPOSE_ENV_FILE=.env.prebuilt docker compose --env-file .env.prebuilt -f docker-compose.services.yml up -d --no-build
```

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8005/health
```

## 7. Run the Compose-Backed Smoke Test Against the Prebuilt Image
The smoke script supports the prebuilt-image path directly:

```bash
COMPOSE_ENV_FILE=.env.prebuilt SMOKE_USE_PREBUILT_IMAGE=1 bash scripts/e2e_smoke.sh
```

Behavior:
- starts MinIO;
- reuses `SERVICE_IMAGE_NAME:SERVICE_IMAGE_TAG` instead of building locally;
- submits ingestion, processing, testset, eval, and reporting jobs;
- validates the end-to-end artifact chain.

## 8. Recommended Tag Selection
- `git-<short_sha>`: best for exact regression reproduction
- `v<VERSION>`: best for release validation
- `dev`: only for short-lived coordination across one CI/build cycle

## 9. Troubleshooting
### 9.1 GHCR pull denied
- confirm package visibility;
- confirm `docker login ghcr.io` succeeded;
- confirm the selected tag exists.

### 9.2 Compose still tries to build
- ensure `SMOKE_USE_PREBUILT_IMAGE=1` is set for the smoke script;
- ensure `docker compose ... up -d --no-build` is used for manual startup;
- ensure `.env.prebuilt` sets both `SERVICE_IMAGE_NAME` and `SERVICE_IMAGE_TAG`.

### 9.3 Wrong owner or tag
The workflow publishes to `ghcr.io/${repository_owner}/rag-eval`. Replace `<owner>` with the repository owner for the CI run that produced the image.

### 9.4 Local parity warning
If [scripts/validate_dev_parity.py](scripts/validate_dev_parity.py) reports a Python mismatch, that reflects the host environment, not the prebuilt image itself. The runtime target remains Python 3.11. Local runs warn by default; CI should use `--python-drift-severity error`.

## 10. Fallback: Internal PyPI Mirror
If prebuilt images are unavailable, use the mirror path documented in [docs/deployment_guide.md](docs/deployment_guide.md) by setting:
- `PIP_INDEX_URL`
- `PIP_EXTRA_INDEX_URL`
- `PIP_TRUSTED_HOST`
- optionally `PIP_NETWORK_CHECK_URL` and `PIP_NETWORK_TIMEOUT`

This is the preferred fallback when your organization provides an internal Python package mirror.
