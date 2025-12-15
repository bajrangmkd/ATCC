# app/db.py
import json
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tenacity import retry, wait_exponential, stop_after_attempt
from app.config import settings

# Quote the password so special characters (like @) don't break the URL parsing.
# If you already url-encoded the password in the .env, quoting again is harmless.
_db_user = settings.DB_USER
_db_pass_quoted = quote_plus(settings.DB_PASS)
_db_host = settings.DB_HOST
_db_port = settings.DB_PORT
_db_name = settings.DB_NAME

# Option A: URL-based engine (safe because password is quoted)
DATABASE_URL = (
    f"mysql+pymysql://{_db_user}:{_db_pass_quoted}"
    f"@{_db_host}:{_db_port}/{_db_name}?charset=utf8mb4"
)

# Option B (alternative, avoids URL encoding entirely — uncomment to use):
# DATABASE_URL = "mysql+pymysql://"
# engine = create_engine(
#     DATABASE_URL,
#     pool_pre_ping=True,
#     pool_size=5,
#     max_overflow=10,
#     connect_args={
#         "host": _db_host,
#         "port": int(_db_port),
#         "user": _db_user,
#         "password": settings.DB_PASS,
#         "db": _db_name,
#         "charset": "utf8mb4",
#     },
# )

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
def test_connection():
    """Try a simple query to validate DB connection. Retries with exponential backoff."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def create_tables_if_not_exists():
    """Create minimal tables required by the app (safe to run repeatedly)."""
    with engine.begin() as conn:
        # create database if missing and then create tables inside it
        conn.execute(text("CREATE DATABASE IF NOT EXISTS atcc"))
        conn.execute(text("USE atcc"))

        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS atcc_cameras (
            camera_id INT AUTO_INCREMENT PRIMARY KEY,
            camera_name VARCHAR(128) NOT NULL,
            rtsp_url TEXT NOT NULL,
            location VARCHAR(255),
            roi JSON NULL,
            fps INT DEFAULT 15,
            enabled TINYINT(1) DEFAULT 1,
            last_seen TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''))

        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS atcc_records (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            detection_id CHAR(36) NOT NULL,
            camera_id INT NOT NULL,
            detected_class VARCHAR(64),
            confidence FLOAT,
            bbox JSON,
            centroid JSON,
            roi_hit TINYINT(1) DEFAULT 0,
            image_path VARCHAR(512),
            passage_time DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
            inference_ms INT,
            extra JSON,
            INDEX (camera_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''))
