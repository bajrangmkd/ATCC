from pydantic import BaseSettings


class Settings(BaseSettings):
DB_HOST: str = 'mysql'
DB_PORT: int = 3306
DB_USER: str
DB_PASS: str
DB_NAME: str = 'atcc'
MODEL_PATH: str = '/models/yolov11.pt'
STORAGE_PATH: str = '/storage'
DETECTION_CONFIDENCE: float = 0.4
CAMERA_POLL_INTERVAL: float = 0.1
CAMERA_CONNECT_TIMEOUT: int = 10
CAMERA_RETRY_BASE: float = 1.0


class Config:
env_file = '/app/.env'


settings = Settings()