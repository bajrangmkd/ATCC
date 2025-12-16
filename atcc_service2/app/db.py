# --- File: app/db.py (Updated and Cleaned) ---
import json
import requests
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tenacity import retry, wait_exponential, stop_after_attempt
from app.config import settings
import uuid 
import traceback 
from typing import List, Dict, Any

# NOTE: VehicleDataPayload and related schemas must now be imported in app/main.py 
# if the external push API is removed from here.

# Quote the password so special characters (like @) don't break the URL parsing.
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

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
def test_connection():
    """Try a simple query to validate DB connection. Retries with exponential backoff."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def create_tables_if_not_exists():
    """Create minimal tables required by the app (safe to run repeatedly)."""
    # Ensure engine is imported and available
    from app.db import engine 
    
    with engine.begin() as conn:
        # Create database if missing and then create tables inside it
        conn.execute(text("CREATE DATABASE IF NOT EXISTS atcc"))
        conn.execute(text("USE atcc"))

        # 1. atcc_cameras table definition (Configuration Table)
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

        # 2. atcc_records table definition (Local Worker Detections)
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS atcc_records (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            detection_id CHAR(36) NOT NULL,
            camera_id INT NOT NULL, 
            camera_name VARCHAR(255) DEFAULT NULL,
            detected_class VARCHAR(64),
            confidence FLOAT,
            bbox JSON,
            centroid JSON,
            roi_hit TINYINT(1) DEFAULT 0,
            image_path VARCHAR(512),
            passage_time DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
            inference_ms INT,
            extra JSON,
            is_sync TINYINT(1) DEFAULT 0,
            sync_status VARCHAR(255) DEFAULT 'pending',
            INDEX (camera_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''))
        
        # 3. atcc_data_api table definition (NEW - Sync Configuration Table)
        # Replaces atcc_sync_targets / atcc_detections_transfer
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS atcc_data_api (
            id INT AUTO_INCREMENT PRIMARY KEY,
            system_name VARCHAR(128) NOT NULL,
            api_url VARCHAR(255) NOT NULL,
            status TINYINT(1) DEFAULT 1, -- 1=True (Enabled), 0=False (Disabled)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''))

        # Ensure changes are saved
        # conn.commit() is handled by engine.begin()
# ----------------------------------------------------------------------


# =====================================================================
# === SYNCHRONIZATION LOGIC (Reads from atcc_data_api) ===
# =====================================================================

SYNC_BATCH_SIZE = 100 

def sync_pending_records() -> dict:
    """
    Fetches pending records, dynamically finds the external API target, 
    sends data, and updates local status upon success, all within a transaction.
    """
    db = SessionLocal()
    records_to_sync = []
    external_sync_url = None

    # 0. FIND ENABLED EXTERNAL API TARGET
    try:
        # Query the newly created configuration table
        target_query = text("""
            SELECT api_url 
            FROM atcc_data_api 
            WHERE status = 1 
            LIMIT 1
        """)
        
        # Use scalar_one_or_none to get the single URL string or None
        external_sync_url = db.execute(target_query).scalar_one_or_none()
        
        if not external_sync_url:
            db.close()
            return {"status": "success", "message": "No enabled external API target found in atcc_data_api."}

        # Check for placeholder endpoint in the URL
        if not external_sync_url.endswith('/'):
            external_sync_url += '/'
        # Append the specific endpoint expected by the external server
        external_sync_url += 'report' 
        
        print(f"Sync target found: {external_sync_url}")

    except Exception:
        db.close()
        print("DB Target Fetch Error:", traceback.format_exc())
        return {"status": "error", "message": "Failed to query external sync target from DB."}


    # 1. FETCH pending records
    try:
        query = text("""
            SELECT 
                id, detection_id, camera_id, camera_name, detected_class, 
                confidence, image_path, passage_time
            FROM 
                atcc_records
            WHERE 
                is_sync = 0 AND sync_status = 'pending'
            LIMIT :batch_size
        """)
        
        result = db.execute(query, {'batch_size': SYNC_BATCH_SIZE})
        
        # Convert fetched rows to a list of dictionaries
        for row in result:
            data = {
                "id": row.id,
                "detection_id": row.detection_id,
                "camera_id": row.camera_id,
                "camera_name": row.camera_name,
                "detected_class": row.detected_class,
                "confidence": row.confidence,
                "image_path": row.image_path,
                "passage_time": row.passage_time
            }
            records_to_sync.append(data)

        if not records_to_sync:
            db.close()
            return {"status": "success", "message": "No pending records found to sync."}
        
    except Exception:
        db.close()
        print("DB Fetch Error:", traceback.format_exc())
        return {"status": "error", "message": "Failed to query pending records from local DB."}

    # 2. PREPARE and PUSH data to external API
    try:
        # Note: We rely on the calling function or external logic to provide a UUID import
        sync_payload = {
            "batch_id": str(uuid.uuid4()), 
            "records": records_to_sync
        }
        
        response = requests.post(external_sync_url, json=sync_payload, timeout=10)
        response.raise_for_status() 

        # External server reported success
        print(f"Successfully pushed batch {sync_payload['batch_id']} of {len(records_to_sync)} records to {external_sync_url}")
        
    except requests.exceptions.RequestException as e:
        db.close()
        print("External Sync Error:", traceback.format_exc())
        return {"status": "warning", "message": f"External push failed to {external_sync_url}. Error: {e}"}

    # 3. UPDATE local records status (Transactional)
    try:
        synced_ids = [r['id'] for r in records_to_sync]
        
        # Use primary key to update status flags
        update_stmt = text("""
            UPDATE 
                atcc_records
            SET 
                is_sync = 1, sync_status = 'synced'
            WHERE 
                id IN :synced_ids
        """).bindparams(synced_ids=tuple(synced_ids))

        db.execute(update_stmt)
        db.commit()
        
        return {
            "status": "success", 
            "message": f"Successfully synced and updated {len(synced_ids)} records to {external_sync_url}.",
            "synced_count": len(synced_ids)
        }
        
    except Exception:
        db.rollback()
        print("DB Update Error (Rollback initiated):", traceback.format_exc())
        return {"status": "error", "message": "External push succeeded, but failed to update local sync status. Rollback executed."}
        
    finally:
        db.close()