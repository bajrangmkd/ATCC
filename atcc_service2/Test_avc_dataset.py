import cv2
import os
import csv
import numpy as np
from ultralytics import YOLO

# Initialize model
model = YOLO(r"D:\ATCC\atcc_service2\models\atcc_model.pt")

# RTSP stream URL
rtsp_url = r"D:\ATCC\atcc_service2\video\video.mp4"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Check RTSP stream
if not cap.isOpened():
    print("❌ Failed to open RTSP stream. Check camera IP, credentials, or firewall settings.")
    exit()

# Setup save paths
os.makedirs("detected_frames", exist_ok=True)
output_csv = "detections.csv"

# Initialize CSV file
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImagePath", "Class", "Confidence"])

# Define ROI Polygon (Edit coordinates as per your frame)
roi_polygon = [(588, 118), (992, 147), (1485, 719), (10, 494)]

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from RTSP stream.")
        break

    # Run YOLO detection
    results = model.predict(source=frame, stream=False, verbose=False)

    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
    confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

    for bbox, cls, conf in zip(detections, classes, confidences):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if cv2.pointPolygonTest(np.array(roi_polygon), (cx, cy), False) >= 0:
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            image_path = os.path.join("detected_frames", f"frame_{cv2.getTickCount()}.jpg")
            cv2.imwrite(image_path, frame)

            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_path, model.names[int(cls)], f"{conf:.2f}"])

    # Draw ROI area
    cv2.polylines(frame, [np.array(roi_polygon)], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()