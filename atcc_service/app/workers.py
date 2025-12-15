# app/workers.py
import os
import time
import uuid
import traceback
import subprocess
from math import floor
from datetime import datetime
from typing import List, Dict, Any
from threading import Event, Lock

# timezone helper
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    LOCAL_TZ = None

def _now_ts_str():
    if LOCAL_TZ is not None:
        dt = datetime.now(LOCAL_TZ)
    else:
        dt = datetime.utcnow()
    return dt.strftime("%Y%m%dT%H%M%S.%f")[:-3], dt

# Lazy imports
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    import app.live as live_mod
except Exception:
    live_mod = None

# inference + optional storage
try:
    from app.inference import predict as run_detection
except Exception:
    def run_detection(frame):
        return [{"label": "car", "confidence": 0.75, "bbox": [10, 10, 200, 100]}]

try:
    from app.storage import save_detection
except Exception:
    save_detection = None

# saving config
SAVE_IMAGE_DIR = os.getenv("ATCC_DETECTIONS_DIR", "data/detections")
SAVE_CROP_DIR = os.getenv("ATCC_DETECTIONS_CROP_DIR", "data/detections/crops")
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_CROP_DIR, exist_ok=True)

RECENT_SAVED: Dict[str, float] = {}
RECENT_SAVED_LOCK = Lock()
SAVE_DEDUP_SECONDS = float(os.getenv("ATCC_DEDUPE_SECONDS", "5.0"))

IN_ROI_STATE: Dict[str, bool] = {}
IN_ROI_LOCK = Lock()

DEBUG_ROI = False

def _coarse_bbox_key(bbox, granularity=8):
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except Exception:
        return None
    return (floor(x1 / granularity), floor(y1 / granularity),
            floor(x2 / granularity), floor(y2 / granularity))

# date-based safe write helpers
def _save_jpeg(frame, camera_id, detection_id, suffix="", quality=92):
    if frame is None:
        return None
    try:
        ts_str, dt = _now_ts_str()
        date_dir = os.path.join(SAVE_IMAGE_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        os.makedirs(date_dir, exist_ok=True)
        fname = f"cam{camera_id}_{ts_str}_{detection_id}{suffix}.jpg"
        path = os.path.join(date_dir, fname)
        if OPENCV_AVAILABLE:
            try:
                ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ret:
                    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                return path
            except Exception:
                try:
                    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return path
                except Exception:
                    print("Failed to write annotated image (cv2):", traceback.format_exc())
                    return None
        return None
    except Exception:
        print("Failed to write annotated image (outer):", traceback.format_exc())
        return None

def _save_crop(frame, bbox, camera_id, detection_id, quality=92):
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
        return None
    except Exception:
        print("Failed to save crop (outer):", traceback.format_exc())
        return None

# ROI helpers (pixel-space normalization)
def _normalize_bbox_to_pixels(bbox, frame_shape):
    if bbox is None:
        return None
    try:
        h, w = frame_shape[0], frame_shape[1]
        vals = list(bbox)
        if len(vals) < 4:
            return None
        vals = [float(v) for v in vals[:4]]
        normalized = all(0.0 <= v <= 1.0 for v in vals)
        if normalized:
            if vals[2] > vals[0] and vals[3] > vals[1]:
                x1 = int(round(vals[0] * w)); y1 = int(round(vals[1] * h))
                x2 = int(round(vals[2] * w)); y2 = int(round(vals[3] * h))
            else:
                x = vals[0]; y = vals[1]; ww = vals[2]; hh = vals[3]
                x1 = int(round(x * w)); y1 = int(round(y * h))
                x2 = int(round((x + ww) * w)); y2 = int(round((y + hh) * h))
        else:
            a,b,c,d = vals
            if (c <= w and d <= h) and (c > 0 and d > 0) and (a + c <= w + 1):
                x1 = int(round(a)); y1 = int(round(b))
                x2 = int(round(a + c)); y2 = int(round(b + d))
            else:
                x1 = int(round(a)); y1 = int(round(b))
                x2 = int(round(c)); y2 = int(round(d))
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        if DEBUG_ROI:
            print(f"_normalize_bbox_to_pixels -> {x1,y1,x2,y2} (frame {w}x{h})")
        return (x1, y1, x2, y2)
    except Exception:
        return None

def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def _bbox_centroid(bbox):
    try:
        x1, y1, x2, y2 = map(float, bbox[:4])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return None

def _bbox_intersects_bbox(b1, b2):
    try:
        ax1, ay1, ax2, ay2 = map(float, b1[:4])
        bx1, by1, bx2, by2 = map(float, b2[:4])
    except Exception:
        return False
    if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
        return False
    return True

def _roi_from_camera_row(camera_row: Dict[str, Any], frame_shape=None):
    roi = camera_row.get("roi")
    if not roi:
        return None
    if isinstance(roi, dict):
        if "bbox" in roi:
            bbox = roi["bbox"]
            if frame_shape is not None:
                bp = _normalize_bbox_to_pixels(bbox, frame_shape)
                if bp:
                    return {"type": "bbox", "bbox": bp}
                return None
            else:
                try:
                    btest = tuple(map(int, bbox[:4])); return {"type":"bbox","bbox":btest}
                except Exception:
                    return None
        if "points" in roi:
            pts = roi["points"]
            if frame_shape is not None:
                out = []; h, w = frame_shape[0], frame_shape[1]
                for p in pts:
                    try:
                        x_f, y_f = float(p[0]), float(p[1])
                    except Exception:
                        continue
                    if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                        out.append((int(round(x_f * w)), int(round(y_f * h))))
                    else:
                        out.append((int(round(x_f)), int(round(y_f))))
                if len(out) >= 3: return {"type":"poly","points":out}
                return None
            else:
                try:
                    out = [(int(p[0]), int(p[1])) for p in pts]
                    if len(out) >= 3: return {"type":"poly","points":out}
                except Exception:
                    pass
                return None
    if isinstance(roi, list):
        pts = roi
        if frame_shape is not None:
            out = []; h, w = frame_shape[0], frame_shape[1]
            for p in pts:
                try:
                    x_f, y_f = float(p[0]), float(p[1])
                except Exception:
                    continue
                if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                    out.append((int(round(x_f * w)), int(round(y_f * h))))
                else:
                    out.append((int(round(x_f)), int(round(y_f))))
            if len(out) >= 3: return {"type":"poly","points":out}
            return None
        else:
            try:
                out = [(int(p[0]), int(p[1])) for p in pts]
                if len(out) >= 3: return {"type":"poly","points":out}
            except Exception:
                pass
    return None

def _roi_contains_detection(roi, bbox_pixels):
    if not roi or not bbox_pixels:
        return False
    try:
        if roi["type"] == "bbox":
            return _bbox_intersects_bbox(roi["bbox"], bbox_pixels)
        if roi["type"] == "poly":
            centroid = _bbox_centroid(bbox_pixels)
            if centroid is None: return False
            return _point_in_poly(centroid[0], centroid[1], roi["points"])
    except Exception:
        return False
    return False

def _normalize_detection_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    label = d.get("label") or d.get("class") or d.get("detected_class")
    confidence = d.get("confidence") or d.get("conf") or d.get("score")
    bbox = d.get("bbox") or d.get("box") or d.get("bounding_box")
    return {"label": label, "confidence": confidence, "bbox": bbox}

# ----------------- Main detection processing (ROI aware) -----------------
def process_detections(detections: List[Dict[str, Any]], camera_row: Dict[str, Any], frame=None):
    cam_id = camera_row.get("camera_id")
    now_ts = time.time()
    saved_records = []
    annotated_draw = None

    frame_shape = frame.shape if frame is not None else None
    roi = _roi_from_camera_row(camera_row, frame_shape=frame_shape)

    for d in detections:
        nd = _normalize_detection_dict(d)
        label = nd.get("label") or "obj"
        conf = nd.get("confidence")
        raw_bbox = nd.get("bbox")

        bbox_px = None
        if raw_bbox is not None:
            bbox_px = _normalize_bbox_to_pixels(raw_bbox, frame_shape if frame_shape is not None else (720,1280))

        coarse = _coarse_bbox_key(bbox_px, granularity=8) if bbox_px is not None else _coarse_bbox_key(raw_bbox, granularity=8)

        if roi is not None:
            if bbox_px is not None:
                inside = _roi_contains_detection(roi, bbox_px)
            else:
                inside = False
        else:
            inside = True

        state_key = f"{cam_id}:{label}:{coarse}"
        trigger_save = False

        if roi is not None:
            with IN_ROI_LOCK:
                prev = IN_ROI_STATE.get(state_key, False)
                if inside and not prev:
                    trigger_save = True
                    IN_ROI_STATE[state_key] = True
                elif not inside and prev:
                    IN_ROI_STATE[state_key] = False
                else:
                    trigger_save = False
        else:
            trigger_save = True

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

        if frame is not None and OPENCV_AVAILABLE:
            try:
                if annotated_draw is None:
                    annotated_draw = frame.copy()

                # draw ROI overlay first (so it's visible under boxes)
                if roi is not None:
                    try:
                        # light transparent fill + border
                        overlay = annotated_draw.copy()
                        if roi["type"] == "poly":
                            pts = roi["points"]
                            if len(pts) >= 3:
                                cv2.fillPoly(overlay, [cv2.convexHull(cv2.array(pts, dtype='int32'))], (0, 128, 0))
                                alpha = 0.15
                                cv2.addWeighted(overlay, alpha, annotated_draw, 1 - alpha, 0, annotated_draw)
                                cv2.polylines(annotated_draw, [cv2.array(pts, dtype='int32')], isClosed=True, color=(0,200,0), thickness=2)
                        elif roi["type"] == "bbox":
                            bx = roi["bbox"]
                            x1,y1,x2,y2 = map(int, bx[:4])
                            cv2.rectangle(annotated_draw, (x1,y1), (x2,y2), (0,200,0), 2)
                    except Exception:
                        # fallback gentle: try drawing raw ROI without alpha if above fails
                        try:
                            if roi.get("type") == "poly":
                                cv2.polylines(annotated_draw, [cv2.array(roi["points"], dtype='int32')], True, (0,200,0), 2)
                            elif roi.get("type") == "bbox":
                                bx = roi["bbox"]; x1,y1,x2,y2 = map(int, bx[:4]); cv2.rectangle(annotated_draw, (x1,y1),(x2,y2),(0,200,0),2)
                        except Exception:
                            pass

                # draw detection bbox (best-effort)
                draw_bbox = bbox_px if bbox_px is not None else None
                if draw_bbox is None and raw_bbox is not None:
                    try:
                        bx = list(map(int, raw_bbox[:4]))
                        if len(bx) >= 4:
                            a,b,c,dv = bx[:4]
                            if c > 0 and dv > 0 and (a + c <= annotated_draw.shape[1] + 1):
                                draw_bbox = (a, b, a + c, b + dv)
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
                if bbox_px is not None:
                    crop_path = _save_crop(frame, bbox_px, cam_id, detection_id)
                else:
                    try:
                        crop_path = _save_crop(frame, raw_bbox, cam_id, detection_id)
                    except Exception:
                        crop_path = None
            except Exception:
                print(f"[cam {cam_id}] frame annotate/store failed:", traceback.format_exc())

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


# ----------------- capture helpers (existing; unchanged) -----------------
def open_rtsp_capture(rtsp_url: str, timeout_s: int = 8, retries: int = 3):
    if not OPENCV_AVAILABLE:
        return None
    for attempt in range(1, retries + 1):
        try:
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
    if cap is None:
        return None
    try:
        ret, frame = cap.read()
    except Exception:
        return None
    if not ret or frame is None:
        return None
    return frame

# FFMPEG fallback helpers (improved)
def start_ffmpeg_process(rtsp_url: str, width: int = 1280, height: int = 720):
    cmd = [
        "ffmpeg", "-rtsp_transport", "tcp", "-stimeout", "5000000",
        "-i", rtsp_url, "-loglevel", "warning",
        "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
        "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "-vf", f"scale={width}:{height}", "-"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        return proc
    except FileNotFoundError:
        return None
    except Exception:
        return None

def read_frame_from_ffmpeg(proc, width: int = 1280, height: int = 720, timeout: float = 6.0):
    try:
        import numpy as np
    except Exception:
        return None
    if proc is None or proc.stdout is None:
        return None
    frame_size = width * height * 3
    t0 = time.time()
    data = b""
    while len(data) < frame_size:
        if proc.poll() is not None:
            return None
        try:
            chunk = proc.stdout.read(frame_size - len(data))
        except Exception:
            chunk = None
        if not chunk:
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

# ----------------- Worker entrypoint (existing; unchanged) -----------------
def process_camera_row(camera_row: Dict[str, Any], stop_event: Event):
    cam_id = camera_row.get("camera_id")
    rtsp = camera_row.get("rtsp_url")
    print(f"Worker started for camera {cam_id}. RTSP={bool(rtsp)}. Test-mode fallback if needed.")

    cap = None
    ff_proc = None
    ff_width, ff_height = 1280, 720
    last_open_attempt = 0
    use_ffmpeg = False

    try:
        while not stop_event.is_set():
            frame = None
            if rtsp and OPENCV_AVAILABLE and not use_ffmpeg:
                if cap is None or not cap.isOpened():
                    if time.time() - last_open_attempt > 2:
                        last_open_attempt = time.time()
                        cap = open_rtsp_capture(rtsp, timeout_s=6, retries=1)
                        if cap is None:
                            use_ffmpeg = True
                            time.sleep(1.0)
                if cap is not None and cap.isOpened():
                    frame = read_frame_from_capture(cap)
                    if frame is None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        use_ffmpeg = True

            if rtsp and (not OPENCV_AVAILABLE or use_ffmpeg):
                if ff_proc is None:
                    ff_proc = start_ffmpeg_process(rtsp, width=ff_width, height=ff_height)
                    if ff_proc is None:
                        ff_proc = None
                        time.sleep(1.0)
                    else:
                        time.sleep(0.2)
                if ff_proc is not None:
                    frame = read_frame_from_ffmpeg(ff_proc, width=ff_width, height=ff_height, timeout=3.0)
                    if frame is None:
                        stop_ffmpeg_process(ff_proc)
                        ff_proc = None
                        if OPENCV_AVAILABLE:
                            use_ffmpeg = False

            if frame is None:
                try:
                    import numpy as np
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                except Exception:
                    frame = None

            try:
                res = run_detection(frame)
                if isinstance(res, tuple) and len(res) == 2:
                    detections, latency_ms = res
                else:
                    detections = res
            except Exception as e:
                print(f"[cam {cam_id}] run_detection error:", e)
                detections = []

            if detections:
                try:
                    process_detections(detections, camera_row, frame=frame)
                except Exception:
                    print(f"[cam {cam_id}] process_detections error:", traceback.format_exc())
                try:
                    annotate_and_store_frame(frame, detections, camera_row)
                except Exception:
                    print(f"[cam {cam_id}] annotate_and_store_frame error:", traceback.format_exc())

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
