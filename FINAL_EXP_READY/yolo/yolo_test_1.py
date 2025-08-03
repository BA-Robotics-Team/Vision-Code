import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = '/home/bestautomation/Downloads/yolo11n.pt'
INFERENCE_SIZE = 640
PERSON_CLASS_ID = 0  # YOLO class ID for 'person'

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Load YOLO model
model = YOLO(MODEL_PATH)

# Inference loop
while True:
    frame = picam2.capture_array()
    rgb_frame = frame[:, :, :3]
    resized = cv2.resize(rgb_frame, (INFERENCE_SIZE, INFERENCE_SIZE))

    results = model.predict(resized, imgsz=INFERENCE_SIZE, verbose=False)

    h_orig, w_orig = rgb_frame.shape[:2]
    scale_x = w_orig / INFERENCE_SIZE
    scale_y = h_orig / INFERENCE_SIZE

    person_count = 0

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            if cls == PERSON_CLASS_ID:
                person_count += 1

            x, y, w, h = box.xywh[0].cpu().numpy()
            conf = box.conf[0].item()
            label = f"{model.names[cls]}: {conf:.2f}"

            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw person count
    cv2.putText(frame, f'Person Count: {person_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print(f'Persons detected: {person_count}', end='\r')

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()


#sudo systemctl stop yolo_autostart.service

