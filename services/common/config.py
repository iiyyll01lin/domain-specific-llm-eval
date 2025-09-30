from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SENSITIVE_FIELDS = {"object_store_access_key", "object_store_secret_key"}
CRITICAL_FIELDS = (
    "object_store_endpoint",
    "object_store_access_key",
    "object_store_secret_key",
    "object_store_bucket",
)
_CONFIG_LOGGED = False


class Settings(BaseSettings):
    service_name: str = Field(default="unknown-service", validation_alias="SERVICE_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    port: int = Field(default=8000, validation_alias="PORT")
    ingestion_db_path: str = Field(default="data/ingestion_jobs.db", validation_alias="INGESTION_DB_PATH")
    processing_db_path: str = Field(default="data/processing_jobs.db", validation_alias="PROCESSING_DB_PATH")
    testset_db_path: str = Field(default="data/testset_jobs.db", validation_alias="TESTSET_DB_PATH")
    eval_db_path: str = Field(default="data/eval_runs.db", validation_alias="EVAL_DB_PATH")
    processing_embedding_max_batch_size: int = Field(
        default=512,
        validation_alias="PROCESSING_EMBEDDING_MAX_BATCH_SIZE",
        ge=1,
    )
    object_store_endpoint: str = Field(..., validation_alias="OBJECT_STORE_ENDPOINT", min_length=1)
    object_store_region: str = Field(default="us-east-1", validation_alias="OBJECT_STORE_REGION", min_length=1)
    object_store_access_key: str = Field(..., validation_alias="OBJECT_STORE_ACCESS_KEY", min_length=1)
    object_store_secret_key: str = Field(..., validation_alias="OBJECT_STORE_SECRET_KEY", min_length=1)
    object_store_use_ssl: bool = Field(default=True, validation_alias="OBJECT_STORE_USE_SSL")
    object_store_bucket: str = Field(..., validation_alias="OBJECT_STORE_BUCKET", min_length=1)
    object_store_max_attempts: int = Field(default=3, validation_alias="OBJECT_STORE_MAX_ATTEMPTS", ge=1)
    object_store_backoff_seconds: float = Field(
        default=0.5,
        validation_alias="OBJECT_STORE_BACKOFF_SECONDS",
        ge=0.0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


def _log_validation_error(exc: ValidationError) -> None:
    missing_fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors() if error["type"] == "missing"]
    if missing_fields:
        logger.error("Missing required configuration values: %s", ", ".join(sorted(missing_fields)))
    else:
        logger.error("Configuration validation error: %s", exc)


def load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:  # pragma: no cover - re-raised for visibility in tests
        _log_validation_error(exc)
        raise RuntimeError("Configuration validation failed") from exc


settings = load_settings()


def _redact_settings_snapshot(raw: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in SENSITIVE_FIELDS and value not in (None, ""):
            snapshot[key] = "***redacted***"
        else:
            snapshot[key] = value
    return snapshot


def log_configuration(current: Optional[Settings] = None) -> None:
    global _CONFIG_LOGGED
    if _CONFIG_LOGGED:
        return
    active = current or settings
    logger.info("Loaded service configuration", extra={"config": _redact_settings_snapshot(active.model_dump())})
    _CONFIG_LOGGED = True


def _ensure_required_fields(required_fields: Iterable[str]) -> None:
    missing = [field for field in required_fields if not getattr(settings, field, None)]
    if missing:
        raise RuntimeError(f"Missing required settings: {', '.join(sorted(missing))}")


def configure_service(service_name: str, *, required_fields: Optional[Iterable[str]] = None) -> Settings:
    """Update the shared settings instance with the active service name and validate requirements."""

    settings.service_name = service_name
    if required_fields:
        _ensure_required_fields(required_fields)
    else:
        _ensure_required_fields(CRITICAL_FIELDS)
    log_configuration(settings)
    return settings


def _reset_config_logging_for_tests() -> None:  # pragma: no cover - used in tests only
    global _CONFIG_LOGGED
    _CONFIG_LOGGED = False


__all__ = [
    "Settings",
    "settings",
    "configure_service",
    "load_settings",
    "log_configuration",
    "CRITICAL_FIELDS",
]
