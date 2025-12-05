# atcc_service/app/inference.py
import os
import time
import random
from typing import List, Dict, Tuple, Any
from app.config import settings


# This file implements a lightweight stub inference. If you add a real YOLOv11 model
# (torch), adapt the _load_real_model() and DummyModel.predict() accordingly.


class DummyModel:
    """A tiny wrapper that either loads a real model (if available) or
    provides randomized dummy detections for testing."""
    def __init__(self) -> None:
        self.model_path = settings.MODEL_PATH
        self.model = None
        self.loaded = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load a real model if the file exists and torch is available."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Attempt to import torch and load the model. If anything fails, fall back.
                try:
                    import torch  # type: ignore
                    # adjust loading logic as needed for your real model format
                    self.model = torch.load(self.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                    self.loaded = True
                except Exception:
                    # If torch isn't installed or loading fails, remain in dummy mode
                    self.model = None
                    self.loaded = False
            else:
                self.loaded = False
        except Exception:
            self.model = None
            self.loaded = False

    def _run_real_inference(self, frame: Any) -> Tuple[List[Dict], int]:
        """Run inference with the loaded model. Adapt to your model's API.
        This implementation is defensive and will fall back to dummy detections on error."""
        start = time.time()
        try:
            # Example placeholder: adapt to your model's predict API.
            # If your model has a .predict or .__call__, call it here and convert output.
            if self.model is None:
                raise RuntimeError("No model loaded")
            # Example (pseudo):
            # raw = self.model(frame)
            # dets = convert_raw_to_standard_format(raw)
            # For now just return dummy if real inference is not implemented.
            raise NotImplementedError("Real inference path not implemented; falling back to dummy")
        except Exception:
            # On any failure, fall back to dummy
            return self._dummy_predict(frame, start)

    def _dummy_predict(self, frame: Any, start_time: float = None) -> Tuple[List[Dict], int]:
        """Generate randomized detections for testing."""
        if start_time is None:
            start_time = time.time()
        # try to read shape, fallback if not available
        try:
            h, w = frame.shape[:2]
        except Exception:
            # fallback to an arbitrary size
            w, h = 1280, 720

        dets: List[Dict] = []
        for _ in range(random.randint(0, 2)):
            x1 = random.uniform(0, w * 0.6)
            y1 = random.uniform(0, h * 0.6)
            x2 = x1 + random.uniform(50, w * 0.4)
            y2 = y1 + random.uniform(30, h * 0.4)
            cls = random.choice(['car', 'truck', 'bus', 'motorbike', 'auto'])
            conf = round(random.uniform(0.45, 0.99), 3)
            dets.append({
                'class': cls,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(min(x2, w - 1)), int(min(y2, h - 1))]
            })
        latency_ms = int((time.time() - start_time) * 1000)
        return dets, latency_ms

    def predict(self, frame: Any) -> Tuple[List[Dict], int]:
        """Public inference entrypoint.
        Returns (detections:list[dict], latency_ms:int)."""
        if self.loaded:
            # attempt real inference; fall back to dummy on failure
            return self._run_real_inference(frame)
        else:
            return self._dummy_predict(frame)


# singleton instance used by the module-level predict shim
MODEL = DummyModel()


def predict(frame: Any) -> Tuple[List[Dict], int]:
    """Module-level helper so other code can do: from app.inference import predict"""
    return MODEL.predict(frame)
