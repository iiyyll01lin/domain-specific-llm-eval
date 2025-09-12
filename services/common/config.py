from pydantic import BaseSettings

class Settings(BaseSettings):
    service_name: str = "unknown-service"
    log_level: str = "INFO"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
