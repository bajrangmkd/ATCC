# --- File: app/workers.py (FULL UPDATE) ---
# This file contains the process_camera_row function and all its dependencies.

import os
import time
import uuid
import traceback
import subprocess
from math import floor
from datetime import datetime
from typing import List, Dict, Any
from threading import Event, Lock
import numpy as np
import json 
# --- NEW: Import settings for default FPS ---
# NOTE: Ensure app.config is reachable
try:
    from app.config import settings
except ImportError:
    # Fallback if config is not properly set up
    class Settings:
        CAMERA_FPS = 10
    settings = Settings()

# timezone helper
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Kolkata")
except ImportError:
    # Fallback to fixed UTC offset if zoneinfo is not available 
    from datetime import timedelta
    LOCAL_TZ = timezone(timedelta(hours=5, minutes=30), name='IST')

def _now_ts_str():
    """
    Returns the current local time (Asia/Kolkata) in three forms required by the worker: 
    1. ts_filename (str): IST time in YYYYMMDDTHHMMSS.mmm format (for unique file names).
    2. ts_db (str): IST time in YYYY-MM-DD HH:MM:SS.ms format (for database insertion).
    3. dt_local (datetime): The datetime object for internal file path construction/calculations.
    """
    
    # 1. Get current time, localized to IST/Asia/Kolkata
    if hasattr(LOCAL_TZ, 'tzname'):
        dt_local = datetime.now(LOCAL_TZ)
    else:
        dt_local = datetime.now()
        
    # 2. Format for filenames (YYYYMMDDTHHMMSS.mmm)
    ts_filename = dt_local.strftime("%Y%m%dT%H%M%S.%f")[:-3] 
    
    # 3. Format for Database (YYYY-MM-DD HH:MM:SS.ms)
    ts_db = dt_local.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Return all three values
    return ts_filename, ts_db, dt_local # <--- CRITICAL CHANGE

# Lazy imports
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    from app.roi import point_in_polygon
except Exception:
    def point_in_polygon(pt, poly_points): return False

try:
    import app.live as live_mod
except Exception:
    live_mod = None

# inference + optional storage
try:
    from app.inference import predict as run_detection
except Exception:
    # NOTE: This fallback model returns ABSOLUTE pixel values for the 640x360 inference frame
    def run_detection(frame):
        return ([{"label": "car", "confidence": 0.75, "bbox": [10, 10, 200, 100]}] , 0)

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

# State tracker for ROI/Line crossing (used for the count trigger)
IN_ROI_STATE: Dict[str, bool] = {}
IN_ROI_LOCK = Lock()

# --- Unique Detection Counter and Storage ---
UNIQUE_DETECTION_ID_COUNTER = 0
UNIQUE_DETECTION_ID_LOCK = Lock()

# Stores the unique ID assigned to a coarse object key (LEGACY, for persistence/drawing fallback)
COARSE_ID_MAP: Dict[str, int] = {} 
COARSE_ID_MAP_LOCK = Lock()

# Stores the last known absolute pixel bbox and unique ID for short-term tracking across frames
TEMPORAL_MAX_AGE_S = 3.0 
TEMPORAL_TRACKER: Dict[int, Dict[str, Any]] = {}
TEMPORAL_TRACKER_LOCK = Lock()

# *** NEW: Buffer to temporarily store expired tracks for proximity matching ***
LOST_TRACK_BUFFER: Dict[int, Dict[str, Any]] = {} 
LOST_TRACK_BUFFER_LOCK = Lock()
LOST_TRACK_MAX_AGE_S = 1.0 

# NEW CONSTANT: Max distance (in pixels) for a new detection to snap to a lost track
LOST_TRACK_MAX_DIST_PX = 50 

# --- END NEW ---

DEBUG_ROI = True

# --- Inference Resizing Constants ---
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360

# --- Final Bounding Box Size Filters ---
MIN_DETECTION_PIXEL_AREA = 10000 
MAX_DETECTION_AREA_RATIO = 0.80 

# --- NEW CONSTANT: Hardcoded Detection Line ---
LINE_Y_COORDINATE = 600 

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
# *** MODIFIED FUNCTION SIGNATURE AND FILENAME ***
def _save_jpeg(frame, camera_id, detection_id, final_count_id, label, quality=92):
    if frame is None:
        return None
    try:
        # Use a placeholder for ts_db (ignored) and get the actual datetime object (dt)
        ts_str, _, dt = _now_ts_str() 
        date_dir = os.path.join(SAVE_IMAGE_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        
        # *** MODIFIED FILENAME FORMAT: cam{id}_{ts}_ID{unique_id}_{label}.jpg ***
        fname = f"cam{camera_id}_{ts_str}_ID{final_count_id}_{label}.jpg"
        
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

# *** MODIFIED FUNCTION SIGNATURE AND FILENAME ***
def _save_crop(frame, bbox, camera_id, detection_id, final_count_id, label, quality=92):
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
        # Use a placeholder for ts_db (ignored) and get the actual datetime object (dt)
        ts_str, _, dt = _now_ts_str() 
        date_dir = os.path.join(SAVE_CROP_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        
        # *** MODIFIED FILENAME FORMAT: cam{id}_{ts}_ID{unique_id}_{label}_crop.jpg ***
        fname = f"cam{camera_id}_{ts_str}_ID{final_count_id}_{label}_crop.jpg"
        
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
    """
    Converts a bounding box (normalized or absolute) to an absolute pixel bbox [x1, y1, x2, y2] 
    for the given frame_shape.
    """
    if bbox is None:
        return None
    try:
        h, w = frame_shape[0], frame_shape[1]
        vals = list(bbox)
        if len(vals) < 4:
            return None
        vals = [float(v) for v in vals[:4]]
        
        # Check if coordinates look like normalized (0.0 to 1.0)
        normalized = all(0.0 <= v <= 1.0 for v in vals)

        # 1. Handle normalized coordinates
        if normalized:
            # Assuming format: [x1_norm, y1_norm, x2_norm, y2_norm] OR [x_norm, y_norm, w_norm, h_norm]
            if vals[2] > vals[0] and vals[3] > vals[1]:
                # Format is [x1, y1, x2, y2]
                x1 = int(round(vals[0] * w)); y1 = int(round(vals[1] * h))
                x2 = int(round(vals[2] * w)); y2 = int(round(vals[3] * h))
            else:
                # Format is [x, y, w, h]
                x = vals[0]; y = vals[1]; ww = vals[2]; hh = vals[3]
                x1 = int(round(x * w)); y1 = int(round(y * h))
                x2 = int(round((x + ww) * w)); y2 = int(round((y + hh) * h))
        
        # 2. Handle absolute coordinates (already pixels for the full frame)
        else:
            # SIMPLIFIED: Assume [x1, y1, x2, y2] or [x, y, w, h] mapped directly to pixels.
            x1 = int(round(vals[0])); y1 = int(round(vals[1]))
            x2 = int(round(vals[2])); y2 = int(round(vals[3]))
        
        # Ensure bounding box is within frame boundaries
        # *** FIX: Assign coordinates individually and enforce int type ***
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1)))
        y2 = int(max(0, min(y2, h - 1)))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        if DEBUG_ROI:
            pass # print(f"_normalize_bbox_to_pixels -> {x1,y1,x2,y2} (frame {w}x{h})")
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
    """Calculates the center point (x, y) of a pixel bbox (x1,y1,x2,y2)."""
    try:
        x1, y1, x2, y2 = map(float, bbox[:4])
        return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
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

def _bbox_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) of two absolute pixel bounding boxes [x1, y1, x2, y2].
    Returns a value between 0.0 and 1.0.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU by dividing the intersection area by the union area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) 
    
    return iou

def _roi_from_camera_row(camera_row: Dict[str, Any], frame_shape=None):
    roi = camera_row.get("roi")
    
    # --- DEBUG 1: Log initial ROI state ---
    if roi is None:
        return None
        
    # --- FIX: Robustly ensure ROI is a dictionary (handles DB string returns) ---
    if isinstance(roi, str):
        try:
            # Handle potential single quotes vs double quotes issues that might pass JSON.loads
            roi = json.loads(roi.replace("'", '"')) 
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[{camera_row.get('camera_id')}] ROI parsing error: failed to decode JSON string: {e}")
            return None
            
    if not isinstance(roi, dict) and not isinstance(roi, list):
        return None
    
    # If the ROI is directly a list of points (unstructured) treat it as a polygon
    if isinstance(roi, list):
        temp_roi = {"points": roi}
    else:
        temp_roi = roi

    if "bbox" in temp_roi:
        bbox = temp_roi["bbox"]
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
    if "points" in temp_roi:
        pts = temp_roi["points"]
        if frame_shape is not None:
            out = []; h, w = frame_shape[0], frame_shape[1]
            for p in pts:
                try:
                    # FIX: Handle {"x": x, "y": y} format from DB (normalized or absolute)
                    if isinstance(p, dict):
                        x_f, y_f = float(p.get("x", p.get(0))), float(p.get("y", p.get(1)))
                    else: # Assuming it's a list/tuple like [x, y]
                        x_f, y_f = float(p[0]), float(p[1])
                except Exception:
                    continue
                
                # Check for normalized coordinates (0.0 to 1.0)
                if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                    out.append((int(round(x_f * w)), int(round(y_f * h))))
                else:
                    # Assume absolute pixel coordinates if outside 0-1 range
                    out.append((int(round(x_f)), int(round(y_f))))
            if len(out) >= 3:
                return {"type":"poly","points":out}
            return None
        else:
            try:
                # Fallback to list indices (safer if conversion fails above)
                out = [(int(p[0]), int(p[1])) for p in pts]
                if len(out) >= 3: return {"type":"poly","points":out}
            except Exception:
                pass
            return None
            
    return None

def _roi_contains_detection(roi, bbox_pixels):
    if not roi or not bbox_pixels:
        return False
    try:
        if roi["type"] == "bbox":
            # Check for intersection with ROI bbox
            return _bbox_intersects_bbox(roi["bbox"], bbox_pixels)
        if roi["type"] == "poly":
            # Check if centroid is within ROI polygon
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
    latency_ms = d.get("latency_ms") or d.get("inference_ms")
    return {"label": label, "confidence": confidence, "bbox": bbox, "latency_ms": latency_ms}

# ----------------- Main detection processing (ROI aware) -----------------
def process_detections(detections: List[Dict[str, Any]], camera_row: Dict[str, Any], frame=None):
    """
    Process detection list. Assigns a unique ID immediately upon appearance (Birth-to-Death tracking)
    and uses the line-crossing event to trigger a single count for that stable ID.
    """
    global UNIQUE_DETECTION_ID_COUNTER 
    global TEMPORAL_TRACKER
    global LOST_TRACK_BUFFER
    
    # --- GET TIME HERE (Fix for AttributeError and time zone) ---
    ts_filename, ts_db, dt_now = _now_ts_str()
    now_ts = time.time() # Used for temporal tracking calculation
    # -----------------------------------------------------------
    
    cam_id = camera_row.get("camera_id")
    # Removed the second call to _now_ts_str() that was causing the error
    saved_records = []
    annotated_draw = None

    frame_shape = frame.shape if frame is not None else (720, 1280, 3)
    roi = _roi_from_camera_row(camera_row, frame_shape=frame_shape) 
    max_area_px = (frame_shape[0] * frame_shape[1]) * MAX_DETECTION_AREA_RATIO
    
    
    # 1. CLEANUP & BUFFER: Move expired tracks to the LOST buffer and clean the buffer.
    with TEMPORAL_TRACKER_LOCK:
        keys_to_remove = []
        for uid, track in TEMPORAL_TRACKER.items():
            age = now_ts - track.get("timestamp", 0)
            if age > TEMPORAL_MAX_AGE_S:
                keys_to_remove.append(uid)
                # Save expired track to the LOST buffer before deletion
                track["lost_timestamp"] = now_ts 
                LOST_TRACK_BUFFER[uid] = track
        
        for uid in keys_to_remove:
            del TEMPORAL_TRACKER[uid]
    
    # Clean up the LOST buffer
    with LOST_TRACK_BUFFER_LOCK:
        lost_keys_to_remove = [uid for uid, track in LOST_TRACK_BUFFER.items() 
                              if (now_ts - track.get("lost_timestamp", 0)) > LOST_TRACK_MAX_AGE_S]
        for uid in lost_keys_to_remove:
            del LOST_TRACK_BUFFER[uid]
    
    updated_temporal_tracker = {}

    # *** FIX: IoU Threshold for aggressive stability ***
    IOU_THRESHOLD = 0.45 

    for d in detections:
        nd = _normalize_detection_dict(d)
        label = nd.get("label") or "obj"
        conf = nd.get("confidence")
        raw_bbox = nd.get("bbox")
        latency_ms = nd.get("latency_ms", 0)

        bbox_px = None
        if raw_bbox is not None:
            frame_dims = frame_shape if frame_shape is not None else (720,1280)
            bbox_px = _normalize_bbox_to_pixels(raw_bbox, frame_dims)
        
        # --- Bounding Box Size Check ---
        if bbox_px is not None:
            w = bbox_px[2] - bbox_px[0]
            h = bbox_px[3] - bbox_px[1]
            area = w * h
            
            if area < MIN_DETECTION_PIXEL_AREA:
                continue 
            if area > max_area_px:
                continue 

        centroid = _bbox_centroid(bbox_px)
        coarse = _coarse_bbox_key(bbox_px, granularity=8) if bbox_px is not None else _coarse_bbox_key(raw_bbox, granularity=8)
        state_key_coarse = f"{cam_id}:{label}:{coarse}" 

        # --- ROI FILTER CHECK ---
        is_currently_inside_roi = True
        if roi is not None:
            if bbox_px is not None:
                is_currently_inside_roi = _roi_contains_detection(roi, bbox_px) 
            else:
                is_currently_inside_roi = False
        
        if roi is not None and not is_currently_inside_roi:
            # Skip processing if outside ROI
            if live_mod is not None:
                 try:
                    live_mod.broadcast_detection(cam_id, {
                        "detection_id": d.get("detection_id") or str(uuid.uuid4()),
                        "camera_id": cam_id,
                        "label": label,
                        "confidence": float(conf) if conf is not None else None,
                        "bbox": raw_bbox,
                        "image_path": None,
                        "roi_hit": False 
                    })
                 except Exception:
                     pass
            continue 

        # ----------------------------------------------------
        # --- ID ASSOCIATION LOGIC: IoU Tracking & Proximity Snapping ---
        # ----------------------------------------------------
        
        assigned_unique_id = None
        best_iou = 0.0
        
        # 2. Check for overlap with existing tracks (from previous frames OR this frame)
        if bbox_px is not None:
            # Combine previous tracks (TEMPORAL_TRACKER) and tracks already matched 
            # in THIS frame (updated_temporal_tracker) for comprehensive matching.
            combined_tracks = {**TEMPORAL_TRACKER, **updated_temporal_tracker}
            
            for uid, track in combined_tracks.items():
                
                # We skip checking against tracks that belong to a different label
                if track.get("label") != label:
                    continue
                    
                iou = _bbox_iou(bbox_px, track["bbox_px"])
                
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    assigned_unique_id = uid
        
        # --- CRITICAL NEW ID ASSIGNMENT/PROXIMITY CHECK (THE FIX) ---
        if assigned_unique_id is None:
            
            # 3. Check proximity against recently LOST tracks
            if centroid is not None:
                with LOST_TRACK_BUFFER_LOCK:
                    min_dist_sq = LOST_TRACK_MAX_DIST_PX**2
                    closest_lost_id = None
                    
                    for uid, track in LOST_TRACK_BUFFER.items():
                        # Calculate distance squared to avoid slow sqrt()
                        lost_centroid = _bbox_centroid(track["bbox_px"])
                        if lost_centroid is not None:
                            dist_sq = (centroid[0] - lost_centroid[0])**2 + (centroid[1] - lost_centroid[1])**2
                            
                            if dist_sq < min_dist_sq and track.get("label") == label:
                                min_dist_sq = dist_sq
                                closest_lost_id = uid
                                
                    if closest_lost_id is not None:
                        # Snap the new detection to the old ID that just expired/was lost
                        assigned_unique_id = closest_lost_id
                        # Remove from LOST_TRACK_BUFFER since we recovered it
                        del LOST_TRACK_BUFFER[closest_lost_id]
                        print(f"[{cam_id}] ID RECOVERED: {assigned_unique_id} (Proximity Match)")
                        
        # 4. If still no ID, assign a truly NEW ID
        if assigned_unique_id is None:
            with UNIQUE_DETECTION_ID_LOCK:
                UNIQUE_DETECTION_ID_COUNTER += 1
                assigned_unique_id = UNIQUE_DETECTION_ID_COUNTER
            
            # Update the COARSE_ID_MAP for persistence 
            with COARSE_ID_MAP_LOCK:
                COARSE_ID_MAP[state_key_coarse] = assigned_unique_id

        
        # --- IMPROVEMENT: Ensure COARSE_ID_MAP is updated for stable ID tracking ---
        with COARSE_ID_MAP_LOCK:
            COARSE_ID_MAP[state_key_coarse] = assigned_unique_id


        # ----------------------------------------------------
        # --- COUNT TRIGGER LOGIC: Single Count on Line Cross ---
        # ----------------------------------------------------
        
        trigger_save = False
        
        if frame is not None: 
            with IN_ROI_LOCK:
                count_state_key = f"{cam_id}:COUNTED:{assigned_unique_id}" 
                
                already_counted = IN_ROI_STATE.get(count_state_key, False)
                
                bottom_center_y = bbox_px[3] if bbox_px is not None else -1
                has_crossed_line = bottom_center_y >= LINE_Y_COORDINATE
                
                # Condition A: Tracked vehicle crosses the line AND hasn't been counted yet
                if has_crossed_line and not already_counted:
                    trigger_save = True 
                    IN_ROI_STATE[count_state_key] = True 
                    print(f"[{cam_id}] COUNT TRIGGERED for ID: {assigned_unique_id} (First Line Cross)")
                            
        
        # 5. Update the Temporal Tracker for continuity in the next frame
        if assigned_unique_id is not None and bbox_px is not None:
             # *** CRITICAL: Add/Update the track in the current frame's match list ***
             updated_temporal_tracker[assigned_unique_id] = {
                 "bbox_px": bbox_px,
                 "label": label,
                 "timestamp": now_ts
             }

        # Handle non-saved/follow-up frames
        if not trigger_save:
            detection_id = d.get("detection_id") or str(uuid.uuid4())
            display_id = assigned_unique_id 
            
            if live_mod is not None:
                try:
                    roi_hit_status = (bbox_px is not None and bbox_px[3] >= LINE_Y_COORDINATE)
                    live_mod.broadcast_detection(cam_id, {
                        "detection_id": str(display_id), 
                        "camera_id": cam_id,
                        "label": label,
                        "confidence": float(conf) if conf is not None else None,
                        "bbox": raw_bbox,
                        "image_path": None,
                        "roi_hit": roi_hit_status
                    })
                except Exception:
                    pass
            continue

        # If trigger_save is True:
        
        detection_id = d.get("detection_id") or str(uuid.uuid4())
        final_count_id = assigned_unique_id 
        
        # NOTE: Dedupe logic here is now mostly redundant but kept for save prevention based on coarse key.
        dedupe_key = f"{cam_id}:{label}:{coarse}"
        with RECENT_SAVED_LOCK:
            RECENT_SAVED[dedupe_key] = now_ts 

        saved_path = None
        crop_path = None
        
        # Save annotated frame
        if frame is not None and OPENCV_AVAILABLE:
            try:
                if annotated_draw is None:
                    annotated_draw = frame.copy() 

                    if roi is not None:
                        overlay = annotated_draw.copy()
                        if roi["type"] == "poly":
                            pts = roi["points"]
                            if len(pts) >= 3:
                                cv2.fillPoly(overlay, [np.array(pts, dtype='int32').reshape((-1, 1, 2))], (0, 128, 0))
                                alpha = 0.15
                                cv2.addWeighted(overlay, alpha, annotated_draw, 1 - alpha, 0, annotated_draw)
                                cv2.polylines(annotated_draw, [np.array(pts, dtype='int32').reshape((-1, 1, 2))], isClosed=True, color=(0,200,0), thickness=2)
                        elif roi["type"] == "bbox":
                            bx = roi["bbox"]
                            x1,y1,x2,y2 = map(int, bx[:4])
                            cv2.rectangle(annotated_draw, (x1,y1), (x2,y2), (0,200,0), 2)
                        
                        h_draw, w_draw = annotated_draw.shape[:2]
                        line_y = LINE_Y_COORDINATE

                draw_bbox = bbox_px if bbox_px is not None else None
                if draw_bbox is not None:
                    x1, y1, x2, y2 = map(int, draw_bbox[:4])
                    h0, w0 = annotated_draw.shape[:2]
                    x1, y1 = max(0, min(x1, w0-1)), max(0, min(y1, h0-1))
                    x2, y2 = max(0, min(x2, w0-1)), max(0, min(y2, h0-1))
                    cv2.rectangle(annotated_draw, (x1,y1), (x2,y2), (0,0,255), 2) 
                    
                    txt = f"{label} ID:{final_count_id} {float(conf):.2f}" if conf is not None else f"{label} ID:{final_count_id}"
                    cv2.putText(annotated_draw, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2) 

                # *** MODIFIED CALL: Passed final_count_id and label ***
                saved_path = _save_jpeg(annotated_draw, cam_id, detection_id, final_count_id, label)
                
                if bbox_px is not None:
                    # *** MODIFIED CALL: Passed final_count_id and label ***
                    crop_path = _save_crop(frame, bbox_px, cam_id, detection_id, final_count_id, label)
                else:
                    try:
                        # *** MODIFIED CALL: Passed final_count_id and label ***
                        crop_path = _save_crop(frame, raw_bbox, cam_id, detection_id, final_count_id, label)
                    except Exception:
                        crop_path = None
            except Exception:
                print(f"[cam {cam_id}] frame annotate/store failed:", traceback.format_exc())

        if save_detection:
            try:
                save_detection({
                    "detection_id": detection_id,
                    "camera_id": cam_id,
                    "camera_name": camera_row.get("camera_name"), # Correctly retrieving camera_name
                    "detected_class": label,
                    "confidence": conf,
                    "bbox": bbox_px if bbox_px is not None else raw_bbox,
                    "centroid": centroid, 
                    "image_path": saved_path,
                    "roi_hit": True, 
                    "passage_time": ts_db, # CRITICAL FIX: Using the formatted IST string
                    "inference_ms": latency_ms, 
                    "extra": d.get("extra", {}),
                    "unique_count_id": final_count_id 
                })
            except Exception:
                print("save_detection failed:", traceback.format_exc())

        if live_mod is not None:
            try:
                live_mod.broadcast_detection(cam_id, {
                    "detection_id": str(final_count_id), 
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

    # 6. Update the main TEMPORAL_TRACKER after processing all detections in the frame
    with TEMPORAL_TRACKER_LOCK:
        TEMPORAL_TRACKER.update(updated_temporal_tracker)
        
    try:
        # Use local time.time() for cleanup threshold, not dt_now
        cutoff = time.time() - (SAVE_DEDUP_SECONDS * 4) 
        with RECENT_SAVED_LOCK:
            for k, ts in list(RECENT_SAVED.items()):
                if ts < cutoff:
                    del RECENT_SAVED[k]
    except Exception:
        pass

    for det_id, spath, cpath in saved_records:
        print(f"[cam {cam_id}] saved detection image for {det_id}: {spath} crop:{cpath}")


# ----------------- Frame annotation and storage (UNCHANGED) -----------------
def annotate_and_store_frame(frame, detections: List[Dict[str, Any]], camera_row: Dict[str, Any]):
    """
    Draw ROI, detection boxes + labels onto a frame copy and store into app.live.LATEST_FRAMES.
    Only draws detections that are inside the ROI and displays the unique count ID.
    """
    if frame is None or not OPENCV_AVAILABLE:
        return
    try:
        draw = frame.copy()
        frame_shape = frame.shape
        cam_id = camera_row.get("camera_id")

        # --- Draw COMPLEX ROI on the Frame for Live View (Optional Polygon) ---
        roi = _roi_from_camera_row(camera_row, frame_shape=frame_shape)
        
        # Draw ROI
        if roi is not None:
            
            if roi.get('type') == 'poly':
                roi_points = roi.get('points', [])
                if len(roi_points) >= 3:
                    roi_np = np.array(roi_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(draw, [roi_np], isClosed=True, color=(0, 255, 0), thickness=3) 
                    cv2.putText(draw, "ROI", (roi_points[0][0], max(16, roi_points[0][1] - 5)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                 
            elif roi.get('type') == 'bbox':
                bx = roi["bbox"]
                if bx and len(bx) >= 4:
                    x1,y1,x2,y2 = map(int, bx[:4])
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 3) 
                    cv2.putText(draw, "ROI", (x1, max(16, y1-6)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- Draw Detection Line on Live Frame (UNCOMMENTED FOR VISUAL DEBUG) ---
        h_draw, w_draw = draw.shape[:2]
        line_y = LINE_Y_COORDINATE
        if 0 < line_y < h_draw:
             cv2.line(draw, (0, line_y), (w_draw, line_y), (0, 165, 255), 2) 
             cv2.putText(draw, "TRIPWIRE", (10, line_y - 10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # --- Existing Detection Drawing Logic ---
        for d in detections:
            nd = _normalize_detection_dict(d)
            bbox = nd["bbox"]
            label = nd["label"] or "obj"
            conf = nd["confidence"] or 0.0
            
            # CRITICAL: Get the COARSE KEY to retrieve the unique ID
            bbox_px_raw = None
            if bbox is not None:
                bbox_px_raw = _normalize_bbox_to_pixels(bbox, frame_shape)
            
            coarse = _coarse_bbox_key(bbox_px_raw, granularity=8) if bbox_px_raw is not None else None
            state_key = f"{cam_id}:{label}:{coarse}" if coarse is not None else None
            
            unique_id_display = None
            # Fetch the stable ID using the coarse map (updated in process_detections)
            if state_key:
                 with COARSE_ID_MAP_LOCK:
                    unique_id_display = COARSE_ID_MAP.get(state_key)


            # CRITICAL: Ensure bbox is normalized to the full frame size
            bbox_px = _normalize_bbox_to_pixels(bbox, frame_shape)
            
            # --- FILTERING LOGIC FOR LIVE VIEW DRAWING ---
            if roi is not None and bbox_px is not None:
                is_currently_inside_roi = _roi_contains_detection(roi, bbox_px)
                if not is_currently_inside_roi:
                    continue 
            # --- END FILTERING LOGIC ---

            if bbox_px and len(bbox_px) >= 4:
                x1, y1, x2, y2 = map(int, bbox_px[:4])
                
                # Highlight based on whether the bottom-center has crossed the LINE
                bottom_center_y = bbox_px[3] 
                has_crossed_line = bottom_center_y >= LINE_Y_COORDINATE
                
                color = (0, 255, 0) 
                if has_crossed_line: 
                    color = (0, 0, 255) # Red for crossing
                
                h, w = draw.shape[:2]
                x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
                x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)

                # Prepare the text label with the unique ID
                txt_parts = [label]
                if unique_id_display is not None:
                    txt_parts.append(f"ID:{unique_id_display}")
                if conf is not None:
                    txt_parts.append(f"{float(conf):.2f}")
                
                txt = " ".join(txt_parts)
                
                cv2.putText(draw, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) 
        
        # CRITICAL: Store the annotated frame in the live module
        if live_mod is not None:
            try:
                live_mod.LATEST_FRAMES[cam_id] = draw
            except Exception:
                print("storing frame to live_mod failed:", traceback.format_exc())
    except Exception:
        print("frame annotate/store failed:", traceback.format_exc())


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

# ----------------- Worker entrypoint -----------------
def process_camera_row(camera_row: Dict[str, Any], stop_event: Event):
    cam_id = camera_row.get("camera_id")
    rtsp = camera_row.get("rtsp_url")
    print(f"Worker started for camera {cam_id}. RTSP={bool(rtsp)}. Test-mode fallback if needed.")

    # *** NOTE: LOCAL_TEST_VIDEO_PATH is now set to None for live cameras ***
    LOCAL_TEST_VIDEO_PATH = r'D:\ATCC\atcc_service2\video\video.mp4'
    # ----------------------------
    
    cap = None
    ff_proc = None
    ff_width, ff_height = 1280, 720
    last_open_attempt = 0
    use_ffmpeg = False
    
    # --- FIX: Failure tracking for stream reset ---
    consecutive_frame_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10 
    
    # --- NEW FPS LOGIC ---
    # 1. Get FPS from camera row (DB) or fall back to config default (15 FPS)
    target_fps = camera_row.get("fps")
    if not isinstance(target_fps, int) or target_fps <= 0:
        target_fps = settings.CAMERA_FPS
    
    # 2. Calculate sleep time per frame (e.g., 1.0 / 15 = 0.0667s)
    target_sleep_time = 1.0 / max(1, target_fps)
    
    print(f"[{cam_id}] Worker running at target FPS: {target_fps} (Sleep: {target_sleep_time:.4f}s)")
    # --- END NEW FPS LOGIC ---


    # # --- MODIFIED STREAM ACQUISITION LOOP ---
    try:
        while not stop_event.is_set():
            frame = None
            t_start = time.time() 

            # Determine the source URL/path for the current loop iteration
            source = LOCAL_TEST_VIDEO_PATH if LOCAL_TEST_VIDEO_PATH and os.path.exists(LOCAL_TEST_VIDEO_PATH) else rtsp

            # --- 1. CAPTURE LOGIC: Use source path ---
            if source and OPENCV_AVAILABLE and not use_ffmpeg:
                if cap is None or not cap.isOpened():
                    if time.time() - last_open_attempt > 2:
                        last_open_attempt = time.time()
                        
                        # Use open_rtsp_capture, which handles local files too
                        cap = open_rtsp_capture(source, timeout_s=6, retries=1) 
                        
                        if cap is None:
                            use_ffmpeg = True
                            time.sleep(1.0)
                        elif LOCAL_TEST_VIDEO_PATH:
                            print(f"[{cam_id}] Reading from local video file: {source}")
                            # --- IMPORTANT: Ensure video loops for continuous testing ---
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            
                if cap is not None and cap.isOpened():
                    frame = read_frame_from_capture(cap)
                    
                    # If reading local video and it ends, loop back to the start
                    if frame is None and LOCAL_TEST_VIDEO_PATH:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame = read_frame_from_capture(cap)
                    
                    if frame is None:
                        # Handle live stream break/failure
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        use_ffmpeg = True

            # 2. Try FFMPEG Fallback (Only useful for live RTSP; skip for simple video files)
            if source == rtsp and (not OPENCV_AVAILABLE or use_ffmpeg):
                if ff_proc is None:
                    ff_proc = start_ffmpeg_process(rtsp, width=ff_width, height=ff_height)
                if ff_proc is not None:
                    frame = read_frame_from_ffmpeg(ff_proc, width=ff_width, height=ff_height, timeout=3.0)
                    if frame is None:
                        stop_ffmpeg_process(ff_proc)
                        ff_proc = None
                        if OPENCV_AVAILABLE:
                            use_ffmpeg = False


            
            # --- Frame Failure/Success Logic ---
            if frame is None:
                consecutive_frame_failures += 1
                
                # FALLBACK: Generate black frame and LOG the failure
                try:
                    import numpy as np
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    print(f"[cam {cam_id}] Stream read FAILED (x{consecutive_frame_failures}). Using black fallback frame.")
                except Exception:
                    frame = None
                
                # Force full restart if failures accumulate
                if consecutive_frame_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[cam {cam_id}] Max frame failures reached. Forcing full stream reset.")
                    if cap:
                        try: cap.release() 
                        except: pass
                    cap = None
                    if ff_proc:
                        try: stop_ffmpeg_process(ff_proc) 
                        except: pass
                    ff_proc = None
                    use_ffmpeg = False
                    consecutive_frame_failures = 0
            else:
                # Frame successfully read
                consecutive_frame_failures = 0
            
            # --- Detection and Processing ---
            
            # Store original frame before potential resize
            original_frame = frame 
            frame_for_inference = original_frame
            
            # Record original dimensions
            if original_frame is not None:
                h_orig, w_orig = original_frame.shape[:2]
            else:
                h_orig, w_orig = (720, 1280) # Default fallback

            # NEW: Resize frame for faster inference if necessary
            if original_frame is not None and original_frame.shape[1] > INFERENCE_WIDTH:
                frame_for_inference = cv2.resize(original_frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT)) 
            
            # Record inference dimensions
            if frame_for_inference is not None:
                h_inf, w_inf = frame_for_inference.shape[:2]
            else:
                h_inf, w_inf = h_orig, w_orig # Use original if inference failed/skipped


            # Measure time spent in detection/processing
            t_detect_start = time.time()
            
            try:
                # Run detection on the (potentially) smaller frame
                res = run_detection(frame_for_inference)
                t_detect_end = time.time()
                
                if isinstance(res, tuple) and len(res) == 2:
                    detections, latency_ms = res
                else:
                    detections = res
                    latency_ms = int((t_detect_end - t_detect_start) * 1000) 

                print(f"[{cam_id}] Detection Latency: {latency_ms} ms (Target: {target_sleep_time*1000:.0f} ms)")
            
            except Exception as e:
                print(f"[{cam_id}] run_detection error:", e)
                detections = []
                latency_ms = int((time.time() - t_detect_start) * 1000) 
            
            # --- START FIX: EXPLICITLY NORMALIZING BBOXES TO 0.0-1.0 OF ORIGINAL FRAME ---
            if w_inf < w_orig and w_inf > 0 and h_inf > 0:
                for d in detections:
                    raw_bbox = d.get("bbox")
                    if raw_bbox is not None and len(raw_bbox) >= 4:
                        try:
                            # Assume raw_bbox is in format [x1, y1, x2, y2] (absolute pixel values on w_inf/h_inf frame)
                            x1, y1, x2, y2 = map(float, raw_bbox[:4])
                            
                            # 1. Normalize to 0-1 range based on the INFERENCE frame size
                            x1_norm_inf = x1 / w_inf
                            y1_norm_inf = y1 / h_inf
                            x2_norm_inf = x2 / w_inf
                            y2_norm_inf = y2 / h_inf
                            
                            # 2. Replace the raw bbox with the normalized values (0-1)
                            d["bbox"] = [x1_norm_inf, y1_norm_inf, x2_norm_inf, y2_norm_inf]
                            
                        except Exception as e:
                            print(f"[{cam_id}] BBox normalization error for detection {d}: {e}")
            # --- END FIX ---


            for d in detections:
                if 'latency_ms' not in d:
                    d['latency_ms'] = latency_ms

            if detections:
                try:
                    process_detections(detections, camera_row, frame=original_frame)
                except Exception:
                    print(f"[{cam_id}] process_detections error:", traceback.format_exc())

            # Annotate and store the latest frame UNCONDITIONALLY (FIX for blank screen)
            try:
                # CRITICAL: Pass original_frame for high-res live view
                annotate_and_store_frame(original_frame, detections, camera_row)
            except Exception:
                print(f"[{cam_id}] annotate_and_store_frame error:", traceback.format_exc())

            # --- MODIFIED: Dynamic Sleep Logic (to maintain FPS) ---
            t_end = time.time()
            time_spent = t_end - t_start
            time_to_sleep = target_sleep_time - time_spent
            
            if time_to_sleep > 0:
                # Use stop_event.wait for cooperative stop during sleep
                if stop_event.wait(timeout=time_to_sleep):
                    break
            else:
                # If processing took too long, yield briefly without waiting for the full interval
                if stop_event.is_set():
                    break
                time.sleep(0.001) # Yield to other threads/processes


    except Exception:
        print(f"[{cam_id}] process_camera_row crashed:", traceback.format_exc())
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