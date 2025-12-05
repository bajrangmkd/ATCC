import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tenacity import retry, wait_exponential, stop_after_attempt
from app.config import settings


DATABASE_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
def test_connection():
with engine.connect() as conn:
conn.execute(text('SELECT 1'))


def create_tables_if_not_exists():
# minimal SQL to create required tables if not present
with engine.begin() as conn:
conn.execute(text('CREATE DATABASE IF NOT EXISTS atcc'))
conn.execute(text('USE atcc'))
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