# app/main.py
import threading
import time
import traceback
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.live import router as live_router


# NOTE: we intentionally avoid importing app.db at module import time to prevent
# import-time failures. DB interactions are done lazily inside functions/startup.

# global structures to track camera workers
_camera_threads: Dict[int, threading.Thread] = {}
_camera_stop_flags: Dict[int, threading.Event] = {}
_poll_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()

POLL_INTERVAL_SECONDS = 20  # how often to poll DB for camera list changes


def load_enabled_cameras_from_db():
    """
    Returns list of dicts: [{'camera_id': int, 'camera_name': str, 'rtsp_url': str, 'roi': json}, ...]
    """
    cameras = []
    try:
        # lazy import to avoid DB logic at module import time
        from app.db import engine
        stmt = text("SELECT camera_id, camera_name, rtsp_url, roi FROM atcc_cameras WHERE enabled=1")
        with engine.connect() as conn:
            res = conn.execute(stmt)
            for row in res:
                m = getattr(row, "_mapping", None) or row
                try:
                    cam_id = int(m["camera_id"])
                except Exception:
                    # skip bad rows
                    continue
                cameras.append({
                    "camera_id": cam_id,
                    "camera_name": m.get("camera_name"),
                    "rtsp_url": m.get("rtsp_url"),
                    "roi": m.get("roi"),
                })
    except Exception:
        # don't crash startup if DB read fails; return empty and let health show DB error
        print("load_enabled_cameras_from_db error:", traceback.format_exc())
    return cameras


def start_camera_worker(camera: Dict[str, Any]):
    """
    Start a worker thread for the camera if not already running.
    """
    cam_id = camera["camera_id"]
    if cam_id in _camera_threads:
        return  # already running

    stop_event = threading.Event()
    _camera_stop_flags[cam_id] = stop_event

    def worker_wrapper(cam_row, stop_evt: threading.Event):
        """
        Wrapper around process_camera_row to make it stoppable in future if needed.
        """
        try:
            # import here to avoid heavy imports at module import time
            from app.workers import process_camera_row
            # pass the stop event into the worker so it can stop cooperatively
            process_camera_row(cam_row, stop_evt)
        except Exception:
            print(f"worker for camera {cam_row.get('camera_id')} crashed:", traceback.format_exc())

    t = threading.Thread(
        target=worker_wrapper,
        args=(camera, stop_event),
        daemon=True,
        name=f"cam_worker_{cam_id}"
    )
    _camera_threads[cam_id] = t
    t.start()
    print(f"Started worker for camera {cam_id} ({camera.get('camera_name')})")

def stop_camera_worker(camera_id: int):
    """
    Signal to stop a camera worker. Sets the stop event for cooperative shutdown.
    """
    evt = _camera_stop_flags.get(camera_id)
    if evt:
        try:
            evt.set()
        except Exception:
            pass
    if camera_id in _camera_threads:
        print(f"Marked worker {camera_id} to stop")
        try:
            del _camera_threads[camera_id]
        except KeyError:
            pass
    if camera_id in _camera_stop_flags:
        try:
            del _camera_stop_flags[camera_id]
        except KeyError:
            pass


def sync_camera_workers():
    """
    Poll the DB for enabled cameras and start/stop workers to match DB state.
    """
    try:
        cameras = load_enabled_cameras_from_db()
        db_ids = set([c["camera_id"] for c in cameras])
        running_ids = set(_camera_threads.keys())

        # start new workers for cameras not running
        for cam in cameras:
            if cam["camera_id"] not in running_ids:
                start_camera_worker(cam)

        # stop workers for cameras removed or disabled
        for rid in list(running_ids):
            if rid not in db_ids:
                print(f"Camera {rid} disabled or removed -> stopping worker")
                stop_camera_worker(rid)

    except Exception:
        print("sync_camera_workers error:", traceback.format_exc())


def polling_loop():
    """
    Background loop that periodically synchronizes camera workers with DB state.
    """
    print("Starting camera polling loop (interval:", POLL_INTERVAL_SECONDS, "s)")
    while not _shutdown_event.is_set():
        try:
            sync_camera_workers()
        except Exception:
            print("Error in polling_loop:", traceback.format_exc())
        # wait with small sleeps so we can respond to shutdown quickly
        for _ in range(POLL_INTERVAL_SECONDS):
            if _shutdown_event.is_set():
                break
            time.sleep(1)
    print("Camera polling loop exiting")


# ------------------ Lifespan handler (startup/shutdown) ------------------
@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    # --- STARTUP ---
    try:
        print("Startup: testing DB connection and ensuring tables...")
        try:
            # lazy import DB helpers
            from app.db import test_connection, create_tables_if_not_exists
            test_connection()
            print("DB OK")
        except Exception as e:
            print("DB test connection failed at startup:", e)

        try:
            # ensure tables exist
            from app.db import create_tables_if_not_exists
            create_tables_if_not_exists()
            print("Ensured required tables exist")
        except Exception as e:
            print("create_tables_if_not_exists failed:", e)

        # initial sync + spawn polling thread
        sync_camera_workers()

        global _poll_thread
        _poll_thread = threading.Thread(target=polling_loop, daemon=True, name="camera_poller")
        _poll_thread.start()
    except Exception:
        print("Error during startup:", traceback.format_exc())
        # re-raise if you want the app to fail to start:
        raise

    # yield to let FastAPI start serving requests
    yield

    # --- SHUTDOWN ---
    print("Shutdown requested: signaling background threads to stop")
    _shutdown_event.set()
    # set all camera stop flags
    for evt in list(_camera_stop_flags.values()):
        try:
            evt.set()
        except Exception:
            pass

    # optional: give threads a short time to clean up (non-blocking)
    try:
        if _poll_thread is not None and _poll_thread.is_alive():
            _poll_thread.join(timeout=2)
    except Exception:
        pass
    print("Lifespan shutdown complete")


# create FastAPI app with lifespan
app = FastAPI(title="ATCC Service", lifespan=lifespan)


@app.get("/", response_class=JSONResponse)
def root():
    return {"service": "atcc", "status": "running", "camera_workers": len(_camera_threads)}


@app.get("/health", response_class=JSONResponse)
def health():
    """
    Returns a JSON health summary:
    - db: reachable?
    - model: loaded or dummy
    - cameras: list of camera_id currently running (worker threads)
    """
    ok = True
    messages = []
    # DB health
    try:
        from app.db import test_connection
        test_connection()
        messages.append("db_ok")
    except Exception as e:
        ok = False
        messages.append(f"db_err:{str(e)}")

    # model status
    try:
        from app.inference import MODEL
        messages.append("model_loaded" if getattr(MODEL, "loaded", False) else "model_dummy")
    except Exception:
        messages.append("model_unknown")

    # cameras
    try:
        running = list(_camera_threads.keys())
        messages.append(f"camera_workers:{len(running)}")
    except Exception:
        messages.append("camera_workers:err")

    payload = {
        "ok": ok,
        "status": messages,
        "cameras_running": list(_camera_threads.keys())
    }
    return payload

# create FastAPI app with lifespan
app = FastAPI(title="ATCC Service", lifespan=lifespan)

# <-- add this line to register the live stream router you imported above
app.include_router(live_router, prefix="/live")   # leave prefix empty or use "/live" if you prefer