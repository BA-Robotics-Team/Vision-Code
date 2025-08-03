import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import math
import time

# ----------------------------- #
# Helper function for distance #
# ----------------------------- #
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --------------------------------------------- #
# Configuration and Initialization              #
# --------------------------------------------- #

# Start GUI support
cv2.startWindowThread()

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
time.sleep(1)  # Small delay to warm up camera

# Load YOLO model (make sure this path is correct)
model = YOLO("/home/bestautomation/Downloads/yolo11n.pt")

# Person class ID for COCO dataset (YOLOv8)
PERSON_CLASS_ID = 0

# For tracking unique persons
tracked_persons = []
max_distance = 50  # Maximum pixel distance to consider the same person

# -------------------------- #
# Continuous detection loop #
# -------------------------- #

while True:
    # Capture frame
    frame = picam2.capture_array()
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image if needed (YOLO will resize internally too)
    results = model.predict(input_img, verbose=False)

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if cls != PERSON_CLASS_ID:
                continue  # Only process persons

            # Extract bounding box in XYWH format
            x, y, w, h = box.xywh[0].cpu().numpy()
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            center = (x, y)

            # Check if this center is already tracked
            already_tracked = False
            for prev_center in tracked_persons:
                if euclidean_distance(center, prev_center) < max_distance:
                    already_tracked = True
                    break

            if not already_tracked:
                tracked_persons.append(center)

            # Draw bounding box and label
            conf = box.conf[0].item()
            label = f"Person: {conf:.2f}"
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display total persons seen
    cv2.putText(frame, f"Total Persons: {len(tracked_persons)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
