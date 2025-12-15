# app/storage.py
import os
from datetime import datetime, timezone
import cv2
import json # NEW: Needed to serialize Python objects (like lists/tuples) into JSON strings for the MySQL JSON columns.

from app.config import settings
from app.db import SessionLocal # NEW: Imports the session factory for DB transactions.
from app.models import atcc_records # NEW: Imports the SQLAlchemy Table definition for insertion.


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

    Args:
        frame: numpy array (H,W,3) BGR (cv2)
        detection_id: uuid string
        camera_id: integer or string used for folder
        camera_name: optional
        quality: JPEG quality (0-100) lower -> smaller size (default 85)

    Returns:
        relative path to STORAGE_BASE (string) or empty string on failure.
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


# ----------------- NEW: Database Persistence Function (Fix 4: Data Not Storing) -----------------

def save_detection(data: dict):
    """
    Saves a single detection record using SQLAlchemy Core.
    This function fixes the missing persistence logic in the worker.
    """
    # Get a new session from the factory
    db = SessionLocal()
    try:
        # Build the INSERT statement using the atcc_records Table object
        stmt = atcc_records.insert().values(
            detection_id=data.get("detection_id"),
            camera_id=data.get("camera_id"),
            detected_class=data.get("detected_class"),
            confidence=data.get("confidence"),
            # JSON fields must be serialized to string for insertion
            bbox=json.dumps(data.get("bbox")),
            centroid=json.dumps(data.get("centroid")),
            roi_hit=bool(data.get("roi_hit", False)),
            image_path=data.get("image_path"),
            passage_time=data.get("passage_time", datetime.now()), # Use provided time or current time
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