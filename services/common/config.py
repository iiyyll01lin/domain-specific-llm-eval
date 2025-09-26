from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "unknown-service"
    log_level: str = "INFO"
    port: int = 8000
    ingestion_db_path: str = "data/ingestion_jobs.db"
    processing_db_path: str = "data/processing_jobs.db"
    object_store_endpoint: Optional[str] = None
    object_store_region: str = "us-east-1"
    object_store_access_key: Optional[str] = None
    object_store_secret_key: Optional[str] = None
    object_store_use_ssl: bool = True
    object_store_bucket: Optional[str] = None
    object_store_max_attempts: int = 3
    object_store_backoff_seconds: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()  # type: ignore
