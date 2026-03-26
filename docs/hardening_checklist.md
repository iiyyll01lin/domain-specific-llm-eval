# Container Hardening Checklist (TASK-125)

Status: Completed – 2025-09-25  
Owner: platform-secops@team

## Verification Steps
1. **Non-root execution**  
   ```bash
   docker build -t rag-eval:test .
   docker run --rm rag-eval:test id
   ```
   Expected: UID/GID printed as non-zero (default `rag:rag`, 1000:1000).

2. **Layer consolidation & multi-stage**  
   ```bash
   docker history rag-eval:test
   ```
   Expected: Total layers < 12, dependency installation isolated within the builder stage (single RUN instruction) and no build toolchain packages in the runtime stage.

3. **Cache hygiene**  
   - `pip install` invoked with `--no-cache-dir`.
   - `/var/lib/apt/lists` cleared in the same layer as package install.

4. **PyPI mirror & offline guard**  
   - Optional build args (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_TRUSTED_HOST`) allow routing through an internal mirror when required.
   - `PIP_NETWORK_CHECK_URL` + `PIP_NETWORK_TIMEOUT` provide a fast connectivity probe; run `docker build --build-arg PIP_NETWORK_TIMEOUT=3 .` and observe the "PyPI unreachable" log when air-gapped.
   - Offline detection now fails the build explicitly with `PyPI unreachable during image build`; use the internal mirror build args or a prebuilt image from connected CI instead of allowing a partial image.

5. **Configurable model cache**  
   - Build argument `MODELS_CACHE` defaults to `/var/cache/rag-models`.
   - Volume declared for the cache path.
   - Directory owned by service user (`rag`).

6. **Ownership and PATH**  
   - `/app` tree (including `services/`) owned by the service user.
   - `$PATH` includes `/home/rag/.local/bin` for future pip --user installs.

7. **Documentation**  
   - `docs/deployment_guide.md` Section 7 summarises the hardening outcome.

8. **Base image patching**  
   - Verify Dockerfile references `python:3.11-slim-bookworm` (builder and runtime stages).
   - `apt-get upgrade -y --no-install-recommends` executes before package install to ensure latest security updates.

## Compose / Smoke-Test Notes
- `docker-compose.services.yml` now forwards the mirror-related build args into Docker builds and tags the service image with `SERVICE_IMAGE_NAME:SERVICE_IMAGE_TAG` (default `rag-eval:dev`).
- `bash scripts/e2e_smoke.sh` supports two deployment paths:
   1. local build path (default), which now fails fast if Docker cannot reach a package source;
   2. prebuilt image path with `SMOKE_USE_PREBUILT_IMAGE=1`, after pulling a CI-built image.

## Operational Notes
- Override `MODELS_CACHE` during build via `--build-arg MODELS_CACHE=/data/models` to align with external volume mounts.
- For compose deployments, mount the cache directory to persist model assets: 
  ```yaml
  volumes:
    - ./models_cache:${MODELS_CACHE_PATH}
  ```
- Future security scans (TASK-124) will enforce Trivy gating for image vulnerabilities.
