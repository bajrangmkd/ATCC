import os
import time
import random
import uuid
from typing import List, Dict
from app.config import settings


# This file implements a lightweight stub inference. If you add a real YOLOv11 model
# (torch), adapt the load_model() and predict() functions accordingly.


class DummyModel:
def __init__(self):
self.model_path = settings.MODEL_PATH
self.loaded = False
if os.path.exists(self.model_path):
# If user placed a real model and torch is available, you can load here.
try:
import torch
self.model = torch.load(self.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
self.loaded = True
except Exception:
# fallback to dummy
self.loaded = False
else:
self.loaded = False


def predict(self, frame) -> (List[Dict], int):
# frame is a numpy array (H,W,3)
start = time.time()
h, w = frame.shape[:2]
# produce 0..2 random detections for testing
dets = []
for _ in range(random.randint(0,2)):
x1 = random.uniform(0, w*0.6)
y1 = random.uniform(0, h*0.6)
x2 = x1 + random.uniform(50, w*0.4)
y2 = y1 + random.uniform(30, h*0.4)
cls = random.choice(['car','truck','bus','motorbike'])
conf = round(random.uniform(0.45, 0.99), 3)
dets.append({
'class': cls,
'confidence': conf,
'bbox': [int(x1), int(y1), int(min(x2,w-1)), int(min(y2,h-1))]
})
latency_ms = int((time.time() - start) * 1000)
return dets, latency_ms


# singleton
MODEL = DummyModel()


def predict(frame):
return MODEL.predict(frame)