# --- File: app/storage.py (FULL UPDATE) ---
import os
import uuid # Needed for fallbacks, though detection_id should be passed
from datetime import datetime, timezone
import cv2
import json 
from sqlalchemy import text, insert
from app.db import engine, SessionLocal # Imports the engine and session factory
# NOTE: Ensure app.models defines the 'atcc_records' Table object
try:
    from app.models import atcc_records 
except ImportError:
    # Placeholder/Fallback if app.models is not fully set up
    atcc_records = None 
    print("Warning: app.models.atcc_records not imported. DB write attempts will fail.")


from app.config import settings

STORAGE_BASE = settings.STORAGE_PATH or '/storage'
os.makedirs(STORAGE_BASE, exist_ok=True)

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def make_image_filename(camera_id: int, detection_id: str, timestamp: datetime, suffix: str = ""):
    # timestamp should be UTC datetime
    ts = timestamp.strftime("%Y%m%dT%H%M%S.%f")[:-3]  # ms precision e.g. 20251204T103215.123
    if suffix:
        filename = f"{camera_id}_{ts}_{detection_id}_{suffix}.jpg"
    else:
        filename = f"{camera_id}_{ts}_{detection_id}.jpg"
    return filename

def save_full_frame(frame, detection_id: str, camera_id: int = None, camera_name: str = None, quality: int = 85) -> str:
    """
    Save the full frame into date-wise folder structure and return relative stored path.
    """
    try:
        now = datetime.now(timezone.utc)
        # path: /storage/YYYY/MM/DD/<camera_id>/
        folder = os.path.join(
            STORAGE_BASE,
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d"),
            str(camera_id or "unknown")
        )
        _safe_mkdir(folder)

        filename = make_image_filename(camera_id or 0, detection_id, now)
        full_path = os.path.join(folder, filename)

        # cv2.imwrite with JPEG quality param:
        # param: [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

        # return relative path from STORAGE_BASE for DB storage
        rel_path = os.path.relpath(full_path, STORAGE_BASE)
        return rel_path.replace(os.path.sep, '/')
    except Exception as e:
        print("save_full_frame error:", e)
        return ""


# ----------------- Database Persistence Function (Updated for new schema) -----------------

def save_detection(data: dict):
    """
    Saves a single detection record using SQLAlchemy Core, including new sync status fields.
    
    Args structure (Expected from worker.py):
        {..., "camera_id": 1, "camera_name": "MyCam", "passage_time": datetime_obj, ...}
    """
    if atcc_records is None:
        print("DATABASE ABORTED: atcc_records Table object is not available.")
        return

    db = SessionLocal()
    try:
        # Build the INSERT statement using the atcc_records Table object
        stmt = insert(atcc_records).values(
            detection_id=data.get("detection_id") or str(uuid.uuid4()),
            camera_id=data.get("camera_id"),
            camera_name=data.get("camera_name"),
            is_sync=0,                                     # Default: 0 (False)
            sync_status='pending',                         # Default: 'pending'
            # -------------------------
            detected_class=data.get("detected_class"),
            confidence=data.get("confidence"),
            # JSON fields must be serialized to string for insertion
            bbox=json.dumps(data.get("bbox")),
            centroid=json.dumps(data.get("centroid")),
            roi_hit=bool(data.get("roi_hit", False)),
            image_path=data.get("image_path"),
            passage_time=data.get("passage_time"), 
            inference_ms=data.get("inference_ms", 0),
            extra=json.dumps(data.get("extra", {})),
        )
        
        # Execute the insertion statement
        db.execute(stmt)
        
        # Commit the transaction to permanently write the data to the database
        db.commit() 
        
    except Exception as e:
        db.rollback() # Rollback changes if an error occurred
        print(f"DATABASE ERROR: Failed to save record {data.get('detection_id')}. Rolling back. Error: {e}")
    finally:
        # Always close the session to release the connection back to the pool
        db.close()