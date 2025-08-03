import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = '/home/bestautomation/Downloads/best.pt'
INFERENCE_SIZE = 640
PERSON_CLASS_ID = 0  # class ID for 'person'

# Load YOLO model
model = YOLO(MODEL_PATH)

# Create DepthAI pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Start pipeline
with dai.Device(pipeline) as device:
    print("✅ OAK-1 live feed with YOLOv8 started. Press 'q' to exit.")
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        frame = video_queue.get().getCvFrame()
        h_orig, w_orig = frame.shape[:2]

        # Resize for YOLO
        resized = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))

        results = model.predict(resized, imgsz=INFERENCE_SIZE, verbose=False)

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

        # Show person count
        cv2.putText(frame, f'Person Count: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f'Persons detected: {person_count}', end='\r')

        cv2.imshow("OAK-1 + YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
