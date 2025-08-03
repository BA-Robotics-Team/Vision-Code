import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = '/home/bestautomation/Downloads/yolo11n.pt'
INFERENCE_SIZE = 640

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}))  # 4-channel image
picam2.start()

# Load YOLO model
model = YOLO(MODEL_PATH)

# Inference loop
while True:
    # Capture frame
    frame = picam2.capture_array()

    # Convert from 4-channel XRGB8888 to RGB
    rgb_frame = frame[:, :, :3]

    # Resize image to match model input (640x640) — may distort aspect ratio
    resized = cv2.resize(rgb_frame, (INFERENCE_SIZE, INFERENCE_SIZE))

    # Run prediction
    results = model.predict(resized, imgsz=INFERENCE_SIZE, verbose=False)

    # Scale factor between resized image and original
    h_orig, w_orig = rgb_frame.shape[:2]
    scale_x = w_orig / INFERENCE_SIZE
    scale_y = h_orig / INFERENCE_SIZE

    # Draw bounding boxes
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"

            # Rescale box coords back to original frame
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Draw
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show result
    cv2.imshow("YOLOv8 Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
