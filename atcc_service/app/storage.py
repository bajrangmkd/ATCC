# app/storage.py
import os
from datetime import datetime, timezone
import cv2

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
