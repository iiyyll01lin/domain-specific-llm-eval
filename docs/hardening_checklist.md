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

2. **Layer consolidation**  
   ```bash
   docker history rag-eval:test
   ```
   Expected: Total layers < 12, pip install combined under single RUN instruction.

3. **Cache hygiene**  
   - `pip install` invoked with `--no-cache-dir`.
   - `/var/lib/apt/lists` cleared in the same layer as package install.

4. **Configurable model cache**  
   - Build argument `MODELS_CACHE` defaults to `/var/cache/rag-models`.
   - Volume declared for the cache path.
   - Directory owned by service user (`rag`).

5. **Ownership and PATH**  
   - `/app` tree (including `services/`) owned by the service user.
   - `$PATH` includes `/home/rag/.local/bin` for future pip --user installs.

6. **Documentation**  
   - `docs/deployment_guide.md` Section 7 summarises the hardening outcome.

## Operational Notes
- Override `MODELS_CACHE` during build via `--build-arg MODELS_CACHE=/data/models` to align with external volume mounts.
- For compose deployments, mount the cache directory to persist model assets: 
  ```yaml
  volumes:
    - ./models_cache:${MODELS_CACHE_PATH}
  ```
- Future security scans (TASK-124) will enforce Trivy gating for image vulnerabilities.
