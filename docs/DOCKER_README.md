# Docker Developer Workflow

Status: Updated for TASK-120 / TASK-121 / TASK-125 – 2025-09-25

## 1. Quick Start
```bash
make dev
```
The command layers `docker-compose.dev.override.yml` on top of the base compose file to:
- Bind mount the repository into `/app` so code edits are visible instantly.
- Keep the shared `models-cache` volume mounted at `/var/cache/rag-models`.
- Launch each service with `uvicorn --reload` for automatic restart.

Stop the stack with:
```bash
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml down
```

## 2. Hot Reload Expectations
- Edit any `services/**` file → Uvicorn restarts within ~3 seconds.
- The `UVICORN_RELOAD=1` flag is set in the override to make the reload behavior explicit.
- Shared dependencies installed into `/app` remain available because the image is built with non-root ownership (TASK-125).

## 3. Environment Overrides
- Default variables come from `.env.compose`.
- Override the env file by exporting `COMPOSE_ENV_FILE=/path/to/custom.env` before running `make dev`.
- Common overrides:
  - `LOG_LEVEL=DEBUG`
  - `OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317`
  - `MODELS_CACHE_PATH=/var/cache/rag-models`

## 4. Model Cache Persistence
The shared `models-cache` named volume ensures downloaded models survive container restarts. To inspect on the host:
```bash
docker volume inspect domain-specific-llm-eval_models-cache
```
Mount a host directory instead by editing `docker-compose.dev.override.yml` and replacing `models-cache:` with a bind mount path.

## 5. Extensions & Plugins
Extensions live under `extensions/` on the host and are bind mounted into `/extensions` inside each container. The loader in `services/common/plugin_loader.py` automatically discovers any `*.py` files that declare `PLUGIN_KIND` and either a `register()` callable or `PLUGIN_DEFINITION` payload. A sample metric is provided in `extensions/sample_metric.py`—drop in your own files to experiment without rebuilding the image.

- Override the mount location by exporting `EXTENSIONS_DIR=/custom/extensions` before launching compose.
- Run `pytest test_plugin_loader.py` to ensure new plugins are discovered correctly.

## 6. File Watcher Troubleshooting
- Ensure the host filesystem supports inotify events (WSL2/macOS via Docker Desktop works out of the box).
- If reloads lag, adjust `UVICORN_RELOAD_DELAY` in the override service environment.
- For large dependency changes, rebuild the image: `make build` followed by `make dev` (allowed after finishing the current implementation phase).

## 7. Validation Snippet
```bash
docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml config | grep --color=always -- '- /data/yy/domain-specific-llm-eval:/app'
```
Confirms that the bind mount remains active alongside the shared cache volume.
