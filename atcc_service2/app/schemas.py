from pydantic import BaseModel, Field # Ensure Field is imported if used below
from typing import Optional, List
from datetime import datetime


class CameraCreate(BaseModel):
    # ALL lines below MUST be indented
    camera_name: str
    rtsp_url: str
    location: Optional[str] = None
    roi: Optional[dict] = None


class DetectionRecord(BaseModel):
    # ALL lines below MUST be indented
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


# =====================================================================
# === NEW MODELS FOR EXTERNAL DATA TRANSFER API (Check these too!) ===
# =====================================================================

# Model for individual vehicle records inside the 'vehicleData' array
class VehicleRecord(BaseModel):
    # ALL lines below MUST be indented
    vehicleType: str
    count: int
    license: str = Field(default="Null")
    lights: str = Field(default="Null")
    color: str = Field(default="Null")
    direction: str
    # These are stored as strings to exactly match the input format
    date: str 
    time: str 
    isStationary: bool
    make: str = Field(default="Null")
    speed: float 
    axleDistance: float
    noOfAxles: int
    cameraName: str # This should be the camera's location/name


# Model for the overall API payload
class VehicleDataPayload(BaseModel):
    # ALL lines below MUST be indented
    deviceObjectId: str # The device/service ID that sent the data
    cameraId: str       # The unique camera ID used by the device
    vehicleData: List[VehicleRecord]
    time: str           # The device's current time (redundant, but in schema)
    deviceMacAddress: str
    createdDate: str    # The device's overall timestamp
    
    # Check the indentation for the inner Config class as well!
    class Config:
        # ALL lines below MUST be indented relative to the class Config
        schema_extra = {
            "example": {
                # ... contents ...
            }
        }