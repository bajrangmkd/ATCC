# app/main.py
import threading
import time
import traceback
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import db
from app.db import test_connection, create_tables_if_not_exists, engine
from app.workers import process_camera_row
from app.config import settings

# global structures to track camera workers
_camera_threads: Dict[int, threading.Thread] = {}
_camera_stop_flags: Dict[int, threading.Event] = {}
_poll_thread: threading.Thread = None
_shutdown_event = threading.Event()

app = FastAPI(title="ATCC Service")

POLL_INTERVAL_SECONDS = 20  # how often to poll DB for camera list changes


def load_enabled_cameras_from_db():
    """
    Returns list of dicts: [{'camera_id': int, 'camera_name': str, 'rtsp_url': str, 'roi': json}, ...]
    """
    cameras = []
    try:
        with engine.connect() as conn:
            res = conn.execute(
                "SELECT camera_id, camera_name, rtsp_url, roi FROM atcc_cameras WHERE enabled=1"
            )
            for row in res:
                cameras.append({
                    "camera_id": int(row["camera_id"]),
                    "camera_name": row["camera_name"],
                    "rtsp_url": row["rtsp_url"],
                    "roi": row["roi"],
                })
    except Exception:
        # don't crash startup if DB read fails; return empty and let health show DB error
        app.logger = getattr(app, "logger", None)
        print("load_enabled_cameras_from_db error:", traceback.format_exc())
    return cameras


def start_camera_worker(camera: Dict[str, Any]):
    """
    Start a worker thread for the camera if not already running.
    Worker threads are daemon threads so they do not block shutdown.
    """
    cam_id = camera["camera_id"]
    if cam_id in _camera_threads:
        return  # already running

    stop_event = threading.Event()
    _camera_stop_flags[cam_id] = stop_event

    def worker_wrapper(cam_row, stop_evt: threading.Event):
        """
        Wrapper around process_camera_row to make it stoppable in future if needed.
        Current process_camera_row loops indefinitely; if you implement stop support
        in workers, check stop_evt periodically.
        """
        try:
            # current process_camera_row is blocking and loops forever.
            # If you implement stop checks inside process_camera_row, pass stop_evt to it.
            process_camera_row(cam_row)
        except Exception:
            print(f"worker for camera {cam_row.get('camera_id')} crashed:", traceback.format_exc())

    t = threading.Thread(target=worker_wrapper, args=(camera, stop_event), daemon=True, name=f"cam_worker_{cam_id}")
    _camera_threads[cam_id] = t
    t.start()
    print(f"Started worker for camera {cam_id} ({camera.get('camera_name')})")


def stop_camera_worker(camera_id: int):
    """
    Signal to stop a camera worker. Currently we only set a stop flag.
    If your worker supports termination, check the corresponding Event to stop it gracefully.
    """
    # mark stop flag so worker can break if it respects the Event
    evt = _camera_stop_flags.get(c_
