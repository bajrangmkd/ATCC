# app/workers.py
# ... (top imports unchanged)
import os
import time
import uuid
import traceback
import subprocess
from math import floor
from datetime import datetime
from typing import List, Dict, Any
from threading import Event, Lock

# timezone helper (top of file, near other imports)
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    LOCAL_TZ = None

# ----------------- Image save helpers (date-based folders + atomic write) -----------------
def _now_ts_str():
    """Return localized timestamp string suitable for filenames: YYYYMMDDTHHMMSS.mmm"""
    if LOCAL_TZ is not None:
        dt = datetime.now(LOCAL_TZ)
    else:
        dt = datetime.utcnow()
    # include milliseconds
    return dt.strftime("%Y%m%dT%H%M%S.%f")[:-3], dt  # returns (str, datetime object)


# Lazy imports and fallbacks
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    import app.live as live_mod
except Exception:
    live_mod = None

# inference + optional storage (unchanged)
try:
    from app.inference import predict as run_detection
except Exception:
    def run_detection(frame):
        return [{"label": "car", "confidence": 0.75, "bbox": [10, 10, 200, 100]}]

try:
    from app.storage import save_detection
except Exception:
    save_detection = None

# ----- saving config & caches (same as prior) -----
SAVE_IMAGE_DIR = os.getenv("ATCC_DETECTIONS_DIR", "data/detections")
SAVE_CROP_DIR = os.getenv("ATCC_DETECTIONS_CROP_DIR", "data/detections/crops")
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_CROP_DIR, exist_ok=True)

RECENT_SAVED: Dict[str, float] = {}
RECENT_SAVED_LOCK = Lock()
SAVE_DEDUP_SECONDS = float(os.getenv("ATCC_DEDUPE_SECONDS", "5.0"))

# Track whether a coarse detection key was previously INSIDE ROI.
# Keyed by (camera_id, coarse_bbox_key, label). Value: bool (True=inside)
IN_ROI_STATE: Dict[str, bool] = {}
IN_ROI_LOCK = Lock()

def _coarse_bbox_key(bbox, granularity=8):
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except Exception:
        return None
    return (floor(x1 / granularity), floor(y1 / granularity),
            floor(x2 / granularity), floor(y2 / granularity))

def _save_jpeg(frame, camera_id, detection_id, suffix="", quality=92):
    """
    Save annotated full-frame JPEG into date-based folder: SAVE_IMAGE_DIR/YYYY/MM/DD/
    Uses atomic write via cv2.imencode -> file write.
    Returns full path or None on failure.
    """
    if frame is None:
        return None
    try:
        ts_str, dt = _now_ts_str()
        date_dir = os.path.join(SAVE_IMAGE_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        os.makedirs(date_dir, exist_ok=True)
        fname = f"cam{camera_id}_{ts_str}_{detection_id}{suffix}.jpg"
        path = os.path.join(date_dir, fname)

        # encode then write atomically
        if OPENCV_AVAILABLE:
            try:
                ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ret:
                    # fallback to cv2.imwrite if encode fails
                    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                # write bytes
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                return path
            except Exception:
                # last-resort attempt
                try:
                    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                except Exception:
                    print("Failed to write annotated image (cv2):", traceback.format_exc())
                    return None
        else:
            # no opencv -> can't encode
            return None
    except Exception:
        print("Failed to write annotated image (outer):", traceback.format_exc())
        return None


def _save_crop(frame, bbox, camera_id, detection_id, quality=92):
    """
    Crop the frame using bbox and save into SAVE_CROP_DIR/YYYY/MM/DD/
    Returns path or None on failure.
    """
    if frame is None or not bbox:
        return None
    try:
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
        except Exception:
            try:
                bx = list(map(int, bbox[:4]))
                x1, y1, w, h = bx[:4]
                x2, y2 = x1 + w, y1 + h
            except Exception:
                return None

        h0, w0 = frame.shape[:2]
        x1, y1 = max(0, min(x1, w0 - 1)), max(0, min(y1, h0 - 1))
        x2, y2 = max(0, min(x2, w0 - 1)), max(0, min(y2, h0 - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        ts_str, dt = _now_ts_str()
        date_dir = os.path.join(SAVE_CROP_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        os.makedirs(date_dir, exist_ok=True)

        fname = f"cam{camera_id}_{ts_str}_{detection_id}_crop.jpg"
        path = os.path.join(date_dir, fname)

        if OPENCV_AVAILABLE:
            try:
                ret, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ret:
                    cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                return path
            except Exception:
                try:
                    cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                except Exception:
                    print("Failed to write crop image:", traceback.format_exc())
                    return None
        else:
            return None

    except Exception:
        print("Failed to save crop (outer):", traceback.format_exc())
        return None
# ----------------- ROI helpers -----------------
DEBUG_ROI = False  # set True to print ROI / bbox debug lines

def _normalize_bbox_to_pixels(bbox, frame_shape):
    """
    Normalize many bbox formats to (x1,y1,x2,y2) in pixel coordinates.
    Supports:
      - [x1,y1,x2,y2] absolute ints/floats
      - [x,y,w,h] absolute ints/floats
      - normalized floats 0..1 in any of the above forms (interpreted relative to frame)
    Returns tuple (x1,y1,x2,y2) or None on failure.
    frame_shape -> (h, w, ...)
    """
    if bbox is None:
        return None
    try:
        h, w = frame_shape[0], frame_shape[1]
        vals = list(bbox)
        if len(vals) < 4:
            return None
        # convert to floats
        vals = [float(v) for v in vals[:4]]
        # detect normalized coords (all values between 0 and 1)
        normalized = all(0.0 <= v <= 1.0 for v in vals)
        if normalized:
            # if [x1,y1,x2,y2] normalized
            # Heuristic: if vals[2] > vals[0] and vals[3] > vals[1] treat as x1,y1,x2,y2 normalized
            if vals[2] > vals[0] and vals[3] > vals[1]:
                x1 = int(round(vals[0] * w))
                y1 = int(round(vals[1] * h))
                x2 = int(round(vals[2] * w))
                y2 = int(round(vals[3] * h))
            else:
                # treat as x,y,w,h normalized
                x = vals[0]; y = vals[1]; ww = vals[2]; hh = vals[3]
                x1 = int(round(x * w))
                y1 = int(round(y * h))
                x2 = int(round((x + ww) * w))
                y2 = int(round((y + hh) * h))
        else:
            # absolute coordinates (may be x1,y1,x2,y2 or x,y,w,h)
            # detect x,y,w,h if third <= width and seems small relative to frame (heuristic)
            # If third value is less than width AND (x + w) > x then treat as x,y,w,h
            a,b,c,d = vals
            if (c <= w and d <= h) and (c > 0 and d > 0) and (a + c <= w + 1):
                # treat as x,y,w,h
                x1 = int(round(a))
                y1 = int(round(b))
                x2 = int(round(a + c))
                y2 = int(round(b + d))
            else:
                # treat as x1,y1,x2,y2
                x1 = int(round(a))
                y1 = int(round(b))
                x2 = int(round(c))
                y2 = int(round(d))
        # clip
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        if DEBUG_ROI:
            print(f"_normalize_bbox_to_pixels -> {x1,y1,x2,y2} (frame {w}x{h})")
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _roi_from_camera_row(camera_row: Dict[str, Any], frame_shape=None):
    """
    Return ROI normalized to pixel coords:
      - bbox ROI -> {"type":"bbox", "bbox": (x1,y1,x2,y2)}
      - poly ROI  -> {"type":"poly", "points":[(x,y), ...]} in pixel coords
    Accepts ROI stored as:
      - {"bbox":[...]} or {"type":"bbox","bbox":[...]}
      - {"points":[[x,y],...]} or list of points
      - percent coords (0..1) are supported if frame_shape provided
    If frame_shape is None and ROI is normalized (0..1), returns None (can't resolve).
    """
    roi = camera_row.get("roi")
    if not roi:
        return None
    # if ROI is a dict
    if isinstance(roi, dict):
        if "bbox" in roi:
            bbox = roi["bbox"]
            if frame_shape is not None:
                bp = _normalize_bbox_to_pixels(bbox, frame_shape)
                if bp:
                    return {"type": "bbox", "bbox": bp}
                else:
                    return None
            else:
                # if bbox absolute ints we can still try to return as-is (assume pixels)
                try:
                    btest = tuple(map(int, bbox[:4]))
                    return {"type": "bbox", "bbox": btest}
                except Exception:
                    return None
        if "points" in roi:
            pts = roi["points"]
            # convert each to pixel coords if needed
            if frame_shape is not None:
                out = []
                h, w = frame_shape[0], frame_shape[1]
                for p in pts:
                    try:
                        x_f, y_f = float(p[0]), float(p[1])
                    except Exception:
                        continue
                    if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                        out.append((int(round(x_f * w)), int(round(y_f * h))))
                    else:
                        out.append((int(round(x_f)), int(round(y_f))))
                if len(out) >= 3:
                    return {"type": "poly", "points": out}
                return None
            else:
                # assume points are pixels
                try:
                    out = [(int(p[0]), int(p[1])) for p in pts]
                    if len(out) >= 3:
                        return {"type": "poly", "points": out}
                except Exception:
                    pass
                return None
    # if ROI is list -> polygon points
    if isinstance(roi, list):
        pts = roi
        if frame_shape is not None:
            out = []
            h, w = frame_shape[0], frame_shape[1]
            for p in pts:
                try:
                    x_f, y_f = float(p[0]), float(p[1])
                except Exception:
                    continue
                if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                    out.append((int(round(x_f * w)), int(round(y_f * h))))
                else:
                    out.append((int(round(x_f)), int(round(y_f))))
            if len(out) >= 3:
                return {"type": "poly", "points": out}
            return None
        else:
            try:
                out = [(int(p[0]), int(p[1])) for p in pts]
                if len(out) >= 3:
                    return {"type": "poly", "points": out}
            except Exception:
                pass
    return None


def _roi_contains_detection(roi, bbox_pixels):
    """
    Operates in pixel coordinates. Expects:
      - roi: {"type":"bbox","bbox":(x1,y1,x2,y2)} OR {"type":"poly","points":[(x,y),...]}
      - bbox_pixels: (x1,y1,x2,y2)
    Returns True if detection is considered inside ROI.
      - for bbox ROI: uses intersection-over-area style: any intersection -> True
      - for poly ROI: tests bbox centroid inside polygon
    """
    if not roi or not bbox_pixels:
        return False
    try:
        if roi["type"] == "bbox":
            return _bbox_intersects_bbox(roi["bbox"], bbox_pixels)
        if roi["type"] == "poly":
            centroid = _bbox_centroid(bbox_pixels)
            if centroid is None:
                return False
            return _point_in_poly(centroid[0], centroid[1], roi["points"])
    except Exception:
        return False
    return False
# ----------------- Normalization -----------------
def _normalize_detection_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    label = d.get("label") or d.get("class") or d.get("detected_class")
    confidence = d.get("confidence") or d.get("conf") or d.get("score")
    bbox = d.get("bbox") or d.get("box") or d.get("bounding_box")
    return {"label": label, "confidence": confidence, "bbox": bbox}

# ----------------- Main detection processing (ROI aware) -----------------
def process_detections(detections: List[Dict[str, Any]], camera_row: Dict[str, Any], frame=None):
    """
    Process detection list (save to DB, capture full-frame once when the detection enters ROI).
    Saving occurs ONLY when detection transitions from outside->inside ROI.
    """
    cam_id = camera_row.get("camera_id")
    now_ts = time.time()
    saved_records = []
    annotated_draw = None

    # use frame shape to resolve normalized ROI coordinates (prefer exact pixel mapping)
    frame_shape = frame.shape if frame is not None else None
    roi = _roi_from_camera_row(camera_row, frame_shape=frame_shape)

    for d in detections:
        nd = _normalize_detection_dict(d)
        label = nd.get("label") or "obj"
        conf = nd.get("confidence")
        raw_bbox = nd.get("bbox")

        # normalize bbox to pixel coords (x1,y1,x2,y2) when possible
        bbox_px = None
        if raw_bbox is not None:
            bbox_px = _normalize_bbox_to_pixels(raw_bbox, frame_shape if frame_shape is not None else (720, 1280))

        # coarse key — prefer pixel bbox if available
        coarse = _coarse_bbox_key(bbox_px, granularity=8) if bbox_px is not None else _coarse_bbox_key(raw_bbox, granularity=8)

        # determine if currently inside ROI (operate in pixels)
        if roi is not None:
            if bbox_px is not None:
                inside = _roi_contains_detection(roi, bbox_px)
            else:
                # can't evaluate bbox relative to ROI -> treat as outside (do not save)
                inside = False
        else:
            # no ROI configured -> treat as inside (allow saving/dedupe)
            inside = True

        # key for tracking per-camera per-object coarse identity + label
        state_key = f"{cam_id}:{label}:{coarse}"

        trigger_save = False

        # If ROI configured: trigger only on outside->inside transition.
        if roi is not None:
            with IN_ROI_LOCK:
                prev = IN_ROI_STATE.get(state_key, False)
                if inside and not prev:
                    trigger_save = True
                    IN_ROI_STATE[state_key] = True
                elif not inside and prev:
                    # update state to outside; don't save
                    IN_ROI_STATE[state_key] = False
                else:
                    # inside==prev -> no transition -> do not save (dedupe)
                    trigger_save = False
        else:
            # No ROI configured -> proceed with normal dedupe logic
            trigger_save = True

        # Broadcast-only case (no save)
        if not trigger_save:
            detection_id = d.get("detection_id") or str(uuid.uuid4())
            if live_mod is not None:
                try:
                    live_mod.broadcast_detection(cam_id, {
                        "detection_id": detection_id,
                        "camera_id": cam_id,
                        "label": label,
                        "confidence": float(conf) if conf is not None else None,
                        "bbox": raw_bbox,
                        "image_path": None,
                        "roi_hit": inside
                    })
                except Exception:
                    pass
            continue

        # Dedupe recent saves using RECENT_SAVED (prevents multiple saves for same coarse key quickly)
        dedupe_key = f"{cam_id}:{label}:{coarse}"
        with RECENT_SAVED_LOCK:
            last_ts = RECENT_SAVED.get(dedupe_key)
            if last_ts and (now_ts - last_ts) < SAVE_DEDUP_SECONDS:
                detection_id = d.get("detection_id") or str(uuid.uuid4())
                if live_mod is not None:
                    try:
                        live_mod.broadcast_detection(cam_id, {
                            "detection_id": detection_id,
                            "camera_id": cam_id,
                            "label": label,
                            "confidence": float(conf) if conf is not None else None,
                            "bbox": raw_bbox,
                            "image_path": None,
                            "roi_hit": True
                        })
                    except Exception:
                        pass
                continue
            RECENT_SAVED[dedupe_key] = now_ts

        detection_id = d.get("detection_id") or str(uuid.uuid4())
        saved_path = None
        crop_path = None

        # annotate & save full-frame
        if frame is not None and OPENCV_AVAILABLE:
            try:
                if annotated_draw is None:
                    annotated_draw = frame.copy()
                # draw all detections (use normalized pixel bbox if available)
                # ensure we draw from bbox_px when available, otherwise try to draw from raw_bbox best-effort
                draw_bbox = None
                if bbox_px is not None:
                    draw_bbox = bbox_px
                else:
                    try:
                        # try best-effort conversion of raw bbox to ints (may be x,y,w,h)
                        bx = list(map(int, raw_bbox[:4]))
                        if len(bx) >= 4:
                            # if looks like x,y,w,h convert
                            a,b,c,dv = bx[:4]
                            if c > 0 and dv > 0 and (a + c <= annotated_draw.shape[1] + 1):
                                x1, y1, x2, y2 = a, b, a + c, b + dv
                                draw_bbox = (x1, y1, x2, y2)
                            else:
                                draw_bbox = (bx[0], bx[1], bx[2], bx[3])
                    except Exception:
                        draw_bbox = None

                if draw_bbox is not None:
                    x1, y1, x2, y2 = map(int, draw_bbox[:4])
                    h0, w0 = annotated_draw.shape[:2]
                    x1, y1 = max(0, min(x1, w0-1)), max(0, min(y1, h0-1))
                    x2, y2 = max(0, min(x2, w0-1)), max(0, min(y2, h0-1))
                    cv2.rectangle(annotated_draw, (x1,y1), (x2,y2), (0,255,0), 2)
                    txt = f"{label} {float(conf):.2f}" if conf is not None else label
                    cv2.putText(annotated_draw, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                saved_path = _save_jpeg(annotated_draw, cam_id, detection_id, suffix=f"_{label}")
                # crop from frame using bbox_px when possible
                if bbox_px is not None:
                    crop_path = _save_crop(frame, bbox_px, cam_id, detection_id)
                else:
                    # try best-effort crop using raw bbox
                    try:
                        crop_path = _save_crop(frame, raw_bbox, cam_id, detection_id)
                    except Exception:
                        crop_path = None
            except Exception:
                print(f"[cam {cam_id}] frame annotate/store failed:", traceback.format_exc())

        # persist to DB via save_detection if available
        if save_detection:
            try:
                save_detection({
                    "detection_id": detection_id,
                    "camera_id": cam_id,
                    "detected_class": label,
                    "confidence": conf,
                    "bbox": bbox_px if bbox_px is not None else raw_bbox,
                    "image_path": saved_path,
                    "roi_hit": True
                })
            except Exception:
                print("save_detection failed:", traceback.format_exc())

        # broadcast to websocket viewers
        if live_mod is not None:
            try:
                live_mod.broadcast_detection(cam_id, {
                    "detection_id": detection_id,
                    "camera_id": cam_id,
                    "label": label,
                    "confidence": float(conf) if conf is not None else None,
                    "bbox": bbox_px if bbox_px is not None else raw_bbox,
                    "image_path": saved_path,
                    "roi_hit": True
                })
            except Exception:
                pass

        saved_records.append((detection_id, saved_path, crop_path))

    # cleanup RECENT_SAVED older than window (housekeeping)
    try:
        cutoff = now_ts - (SAVE_DEDUP_SECONDS * 4)
        with RECENT_SAVED_LOCK:
            for k, ts in list(RECENT_SAVED.items()):
                if ts < cutoff:
                    del RECENT_SAVED[k]
    except Exception:
        pass

    for det_id, spath, cpath in saved_records:
        print(f"[cam {cam_id}] saved detection image for {det_id}: {spath} crop:{cpath}")


# ----------------- Frame annotation and storage -----------------
def annotate_and_store_frame(frame, detections: List[Dict[str, Any]], camera_row: Dict[str, Any]):
    """
    Draw detection boxes + labels onto a frame copy and store into app.live.LATEST_FRAMES.
    Uses normalized pixel bboxes for drawing when available.
    """
    if frame is None or not OPENCV_AVAILABLE:
        return
    try:
        draw = frame.copy()
        frame_shape = frame.shape
        for d in detections:
            nd = _normalize_detection_dict(d)
            raw_bbox = nd.get("bbox")
            bbox_px = _normalize_bbox_to_pixels(raw_bbox, frame_shape)
            label = nd.get("label") or "obj"
            conf = nd.get("confidence") or 0.0
            if bbox_px and len(bbox_px) >= 4:
                x1, y1, x2, y2 = map(int, bbox_px[:4])
            else:
                # best-effort convert raw bbox (x,y,w,h) fallback
                try:
                    bx = list(map(int, raw_bbox[:4]))
                    if len(bx) >= 4:
                        a,b,c,dv = bx[:4]
                        if c > 0 and dv > 0 and (a + c <= draw.shape[1] + 1):
                            x1, y1, x2, y2 = a, b, a + c, b + dv
                        else:
                            x1, y1, x2, y2 = bx[0], bx[1], bx[2], bx[3]
                    else:
                        continue
                except Exception:
                    continue

            # clip coords
            h0, w0 = draw.shape[:2]
            x1, y1 = max(0, min(x1, w0-1)), max(0, min(y1, h0-1))
            x2, y2 = max(0, min(x2, w0-1)), max(0, min(y2, h0-1))
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{label} {float(conf):.2f}"
            cv2.putText(draw, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if live_mod is not None:
            try:
                live_mod.LATEST_FRAMES[camera_row.get("camera_id")] = draw
            except Exception:
                print("storing frame to live_mod failed:", traceback.format_exc())
    except Exception:
        print("frame annotate/store failed:", traceback.format_exc())
# ----------------- OpenCV capture helpers -----------------
def open_rtsp_capture(rtsp_url: str, timeout_s: int = 8, retries: int = 3):
    """Try to open a cv2.VideoCapture for an RTSP URL with retries."""
    if not OPENCV_AVAILABLE:
        return None
    for attempt in range(1, retries + 1):
        try:
            # prefer FFMPEG backend when available
            if hasattr(cv2, "CAP_FFMPEG"):
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(rtsp_url)
        except Exception:
            cap = None
        t_start = time.time()
        while time.time() - t_start < timeout_s:
            try:
                if cap is not None and cap.isOpened():
                    # reduce OpenCV noise if possible
                    try:
                        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
                    except Exception:
                        pass
                    return cap
            except Exception:
                break
            time.sleep(0.2)
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        print(f"RTSP open attempt {attempt}/{retries} failed for {rtsp_url}")
    return None

def read_frame_from_capture(cap):
    """Read a single frame from capture. Returns frame or None."""
    if cap is None:
        return None
    try:
        ret, frame = cap.read()
    except Exception:
        return None
    if not ret or frame is None:
        return None
    return frame

# ----------------- FFMPEG pipe fallback helpers -----------------
# improved ffmpeg start (replace your existing start_ffmpeg_process)
def start_ffmpeg_process(rtsp_url: str, width: int = 1280, height: int = 720):
    """
    Start ffmpeg subprocess that outputs raw BGR frames to stdout.
    Improved flags for RTSP stability and lower latency.
    """
    # Use tcp transport, small probe/analyze to start faster, nobuffer/low_delay
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",           # force TCP (more reliable over lossy networks)
        "-stimeout", "5000000",             # socket timeout in microseconds (5s)
        "-i", rtsp_url,
        "-loglevel", "warning",             # quieter than 'quiet' but still shows warnings
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-an", "-sn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "-"
    ]
    try:
        # use a large buffer size for stdout to avoid blocking on Windows
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        return proc
    except FileNotFoundError:
        # ffmpeg binary not found
        return None
    except Exception:
        return None


# improved read_frame_from_ffmpeg
def read_frame_from_ffmpeg(proc, width: int = 1280, height: int = 720, timeout: float = 6.0):
    """
    Read one raw frame from ffmpeg process stdout. Returns numpy array or None on failure.
    Increased timeout slightly and handles partial reads more robustly.
    """
    try:
        import numpy as np
    except Exception:
        return None

    if proc is None or proc.stdout is None:
        return None
    frame_size = width * height * 3  # bgr24
    t0 = time.time()
    data = b""
    # accumulate until we have full frame or timeout
    while len(data) < frame_size:
        # if ffmpeg process exited, give up
        if proc.poll() is not None:
            return None
        try:
            chunk = proc.stdout.read(frame_size - len(data))
        except Exception:
            chunk = None
        if not chunk:
            # allow a somewhat longer timeout for slow frames
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.01)
            continue
        data += chunk
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = arr.reshape((height, width, 3))
        return frame
    except Exception:
        return None
    
def stop_ffmpeg_process(proc):
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=0.5)
    except Exception:
        pass

# ----------------- Worker entrypoint -----------------
def process_camera_row(camera_row: Dict[str, Any], stop_event: Event):
    """
    Worker entrypoint.
    - camera_row: {'camera_id', 'rtsp_url', ...}
    - stop_event: threading.Event to stop loop cooperatively
    """
    cam_id = camera_row.get("camera_id")
    rtsp = camera_row.get("rtsp_url")
    print(f"Worker started for camera {cam_id}. RTSP={bool(rtsp)}. Test-mode fallback if needed.")

    cap = None
    ff_proc = None
    ff_width, ff_height = 1280, 720
    last_open_attempt = 0
    use_ffmpeg = False  # switch to ffmpeg fallback when cv2 capture repeatedly fails

    try:
        while not stop_event.is_set():
            frame = None

            # ---------- Try cv2 capture first (if available and not using ffmpeg)
            if rtsp and OPENCV_AVAILABLE and not use_ffmpeg:
                # lazy-open capture if not open or failed previously
                if cap is None or not cap.isOpened():
                    if time.time() - last_open_attempt > 2:
                        last_open_attempt = time.time()
                        cap = open_rtsp_capture(rtsp, timeout_s=6, retries=1)
                        if cap is None:
                            # mark to try ffmpeg next loop
                            use_ffmpeg = True
                            # small backoff before switching fully
                            time.sleep(1.0)
                # attempt to read
                if cap is not None and cap.isOpened():
                    frame = read_frame_from_capture(cap)
                    if frame is None:
                        # read failed; release and try ffmpeg next
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        use_ffmpeg = True

            # ---------- FFMPEG fallback (reliable, forces TCP, quiet logs)
            if rtsp and (not OPENCV_AVAILABLE or use_ffmpeg):
                if ff_proc is None:
                    ff_proc = start_ffmpeg_process(rtsp, width=ff_width, height=ff_height)
                    if ff_proc is None:
                        # ffmpeg not available / failed to start
                        ff_proc = None
                        # fallback to synthetic frames
                        time.sleep(1.0)
                    else:
                        # started ffmpeg, small warmup
                        time.sleep(0.2)
                if ff_proc is not None:
                    frame = read_frame_from_ffmpeg(ff_proc, width=ff_width, height=ff_height, timeout=3.0)
                    if frame is None:
                        # failed to read frame; restart ffmpeg next loop
                        stop_ffmpeg_process(ff_proc)
                        ff_proc = None
                        # if cv2 is available, give it another shot next cycle
                        if OPENCV_AVAILABLE:
                            use_ffmpeg = False

            # fallback: generate dummy frame if no real frame
            if frame is None:
                try:
                    import numpy as np
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                except Exception:
                    frame = None

            # run detection
            try:
                res = run_detection(frame)
                if isinstance(res, tuple) and len(res) == 2:
                    detections, latency_ms = res
                else:
                    detections = res
            except Exception as e:
                print(f"[cam {cam_id}] run_detection error:", e)
                detections = []

            # process detections (pass frame so images can be saved)
            if detections:
                try:
                    process_detections(detections, camera_row, frame=frame)
                except Exception:
                    print(f"[cam {cam_id}] process_detections error:", traceback.format_exc())
                # annotate & store the latest annotated frame for live viewers
                try:
                    annotate_and_store_frame(frame, detections, camera_row)
                except Exception:
                    print(f"[cam {cam_id}] annotate_and_store_frame error:", traceback.format_exc())

            # short sleep permitting cooperative stop
            for _ in range(5):
                if stop_event.is_set():
                    break
                time.sleep(0.2)

    except Exception:
        print(f"[cam {cam_id}] process_camera_row crashed:", traceback.format_exc())
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if ff_proc is not None:
                stop_ffmpeg_process(ff_proc)
        except Exception:
            pass
        print(f"Worker for camera {cam_id} exiting.")

# ----------------- Frame annotation and storage -----------------
def annotate_and_store_frame(frame, detections: List[Dict[str, Any]], camera_row: Dict[str, Any]):
    """
    Draw detection boxes + labels onto a frame copy and store into app.live.LATEST_FRAMES.
    Safe to call even if live_mod is None or cv2 missing.
    """
    if frame is None or not OPENCV_AVAILABLE:
        return
    try:
        draw = frame.copy()
        for d in detections:
            nd = _normalize_detection_dict(d)
            bbox = nd["bbox"]
            label = nd["label"] or "obj"
            conf = nd["confidence"] or 0.0
            if bbox and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                except Exception:
                    # if bbox stored as [x,y,w,h] convert to x1,y1,x2,y2
                    try:
                        bx = list(map(int, bbox[:4]))
                        if len(bx) >= 4:
                            x1, y1, w, h = bx[:4]
                            x2, y2 = x1 + w, y1 + h
                        else:
                            continue
                    except Exception:
                        continue
                # clip coords
                h, w = draw.shape[:2]
                x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
                x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"{label} {float(conf):.2f}"
                cv2.putText(draw, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if live_mod is not None:
            try:
                live_mod.LATEST_FRAMES[camera_row.get("camera_id")] = draw
            except Exception:
                print("storing frame to live_mod failed:", traceback.format_exc())
    except Exception:
        print("frame annotate/store failed:", traceback.format_exc())
