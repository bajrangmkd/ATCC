from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CameraCreate(BaseModel):
camera_name: str
rtsp_url: str
location: Optional[str] = None
roi: Optional[dict] = None


class DetectionRecord(BaseModel):
detection_id: str
camera_id: int
detected_class: str
confidence: float
bbox: dict
centroid: dict
roi_hit: bool
image_path: Optional[str]
passage_time: datetime
inference_ms: int