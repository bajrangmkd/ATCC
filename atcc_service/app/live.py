# app/live.py
from fastapi import APIRouter, Response, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import time
import cv2
import numpy as np

router = APIRouter()

# global store of latest frames annotated by workers
LATEST_FRAMES = {}           # camera_id -> numpy ndarray (BGR)

# helper: encode numpy BGR frame to JPEG bytes
def encode_jpeg_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    if frame is None:
        return None
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()

@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: int):
    """Return a single JPEG snapshot (good quick debug)."""
    frame = LATEST_FRAMES.get(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available yet")
    jpg = encode_jpeg_bytes(frame)
    if jpg is None:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=jpg, media_type="image/jpeg")

def mjpeg_generator(camera_id: int, fps: int = 8):
    boundary = b'--frame'
    interval = 1.0 / max(1, fps)
    while True:
        frame = LATEST_FRAMES.get(camera_id)
        if frame is None:
            # sleep briefly while waiting for a frame
            time.sleep(0.1)
            continue
        jpg = encode_jpeg_bytes(frame)
        if jpg:
            yield boundary + b'\r\n' + b'Content-Type: image/jpeg\r\n' + b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
        else:
            # send a tiny keepalive delay
            time.sleep(0.1)
        time.sleep(interval)

@router.get("/{camera_id}/mjpeg")
def mjpeg(camera_id: int):
    # streaming response: keep the connection open
    return StreamingResponse(mjpeg_generator(camera_id), media_type='multipart/x-mixed-replace; boundary=frame')

# optional html viewer
@router.get("/{camera_id}/view")
def view(camera_id: int):
    html = f"""
    <html><body style="background:#000;color:#fff">
    <h2>Camera {camera_id} — live detection</h2>
    <img src="/live/{camera_id}/mjpeg" style="max-width:100%;height:auto" />
    </body></html>
    """
    return HTMLResponse(content=html)
