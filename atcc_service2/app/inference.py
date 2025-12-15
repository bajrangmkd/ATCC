# atcc_service/app/inference.py
import os
import time
import random
from typing import List, Dict, Tuple, Any
from app.config import settings

# --- NEW IMPORTS ---
try:
    from ultralytics import YOLO # Import the actual YOLO library
except ImportError:
    # If ultralytics is not installed, the service cannot run a real model.
    YOLO = None 

# --- Vehicle Class Filtering ---
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'bike', 'auto','lcv'] 


class Model: # Renamed from DummyModel to Model to hold the new logic
    """A wrapper to load and run the real YOLO model. Fails if model is missing."""
    def __init__(self) -> None:
        self.model_path = settings.MODEL_PATH
        self.model = None
        self.loaded = False
        self._try_load_model()
        
    # --- MODEL LOADING LOGIC (CRASH ON FAILURE) ---
    def _try_load_model(self) -> None:
        """Attempt to load a real YOLO model. Raises RuntimeError on failure."""
        if YOLO is None:
            # If the library is missing, crash immediately.
            raise RuntimeError("FATAL ERROR: The 'ultralytics' library is not installed. Service cannot run.")
            
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Use the path to load the YOLO model
                self.model = YOLO(self.model_path)
                
                if hasattr(self.model, 'names'):
                    print(f"YOLO model loaded from {self.model_path} with {len(self.model.names)} classes.")
                    self.loaded = True
                else:
                    self.loaded = False
            else:
                # If path is incorrect or file is missing, raise error
                raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
        
        # --- MODIFIED: CRASH SERVICE ON FAILURE ---
        except Exception as e:
            error_message = f"FATAL ERROR: Failed to load detection model from {self.model_path}. Service cannot run. Error: {e}"
            print(error_message)
            raise RuntimeError(error_message) from e

    # --- REAL INFERENCE LOGIC (Unchanged) ---
    def _run_real_inference(self, frame: Any) -> Tuple[List[Dict], int]:
        """
        Run inference with the loaded YOLO model and convert output to standard dictionary format.
        """
        start = time.time()
        final_detections: List[Dict] = []
        
        try:
            # This check will only fail if the model was loaded, but then somehow corrupted.
            if self.model is None:
                raise RuntimeError("No model loaded")
                
            results = self.model.predict(source=frame, stream=False, verbose=False)
            
            # Process the results object (assuming batch size 1)
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                min_conf = settings.DETECTION_CONFIDENCE
                
                for bbox, cls_id, conf in zip(boxes, classes, confidences):
                    class_name = self.model.names.get(int(cls_id), "unknown")
                    
                    # --- Apply Confidence and Class Filtering ---
                    if conf >= min_conf and class_name in VEHICLE_CLASSES:
                        final_detections.append({
                            'class': class_name,
                            'class_id': int(cls_id),
                            'confidence': float(conf),
                            'bbox': [float(x) for x in bbox] 
                        })

            latency_ms = int((time.time() - start) * 1000)
            return final_detections, latency_ms

        except Exception as e:
            error_message = f"Inference failed during prediction. Service crash likely. Error: {e}"
            print(error_message)
            raise RuntimeError(error_message) from e
        
    def predict(self, frame: Any) -> Tuple[List[Dict], int]:
        """Public inference entrypoint.
        Returns (detections:list[dict], latency_ms:int)."""
        if self.loaded:
            return self._run_real_inference(frame)
        else:
            # If not loaded, the initializer should have already crashed the service.
            raise RuntimeError("Model is not loaded. Predict function should not be called.")


# singleton instance used by the module-level predict shim
MODEL = Model()


def predict(frame: Any) -> Tuple[List[Dict], int]:
    """Module-level helper so other code can do: from app.inference import predict"""
    return MODEL.predict(frame)