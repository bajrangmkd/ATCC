from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    DB_HOST: str = "mysql"
    DB_PORT: int = 3306
    DB_USER: str = Field(default="root")
    DB_PASS: str = Field(default="Dev@12345")
    DB_NAME: str = "atcc"

    MODEL_PATH: str = "/models/atcc_model.pt"
    STORAGE_PATH: str = "/storage"
    DETECTION_CONFIDENCE: float = 0.4
    CAMERA_POLL_INTERVAL: float = 0.1
    CAMERA_CONNECT_TIMEOUT: int = 10
    CAMERA_RETRY_BASE: float = 1.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
