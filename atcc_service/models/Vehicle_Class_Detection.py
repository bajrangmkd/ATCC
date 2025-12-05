# This model test with RTSP Stream 
import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO(r"D:\workspace\ATCC\runs\detect\ATCC_modelV14\weights\best.pt")

# RTSP stream URL & Video file path
video_url = r"D:\workspace\ATCC\Video\video.mp4"  # Replace with your RTSP stream URL or video file path



# Open video stream
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream. Check camera IP, credentials, or firewall settings.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from RTSP stream.")
        break

    # Run detection
    results = model.predict(source=frame, stream=False, verbose=False)

    # Draw boxes on the frame
    annotated_frame = results[0].plot(2)

    # Display frame
    cv2.imshow("RTSP Stream", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()


# This code for test from images add


# from ultralytics import YOLO

# # Load trained model
# model = YOLO(r"C:\Users\admin\Desktop\Python\runs\detect\train\weights\last.pt")  # Ensure correct path

# # Define class names
# class_names = ['CAR', 'veh_plate']

# # Define image path
# image_path = r"C:\Users\admin\Desktop\Python\dataset object detection\valid\images\LANE14_20250108_180237_jpg.rf.1a3593535ab49592352b4459c42b5cc6.jpg"

# # Run prediction
# results = model.predict(source=image_path, save=True, conf=0.2, imgsz=1280)

# # Process results
# for r in results:
#     for box in r.boxes:
#         class_id = int(box.cls.item())  # Convert to integer properly
#         confidence = float(box.conf.item())  # Convert to float
#         print(f"Detected: {class_names[class_id]} with {confidence:.2f} confidence")

# # Show results
# for r in results:
#     r.show()  # Display image with detections
