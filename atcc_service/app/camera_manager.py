import cv2
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from app.config import settings


# Use OpenCV to open RTSP. Provide backoff on failure.


def open_camera(rtsp_url: str, timeout: int = None):
timeout = timeout or settings.CAMERA_CONNECT_TIMEOUT
cap = cv2.VideoCapture(rtsp_ur