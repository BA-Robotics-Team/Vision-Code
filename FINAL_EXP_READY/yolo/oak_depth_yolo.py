import depthai as dai
import cv2
import json
import numpy as np
from ultralytics import YOLO

# -------- Load COCO Segmentation ROI Polygon --------
with open("/home/bestautomation/Downloads/labels_my-project-name_2025-07-22-01-54-08.json", "r") as f:
    data = json.load(f)

segmentation = data["annotations"][0]["segmentation"][0]
points = np.array(segmentation, dtype=np.float32).reshape((-1, 2))  # P0 → P1 → P2 → P3
drawn_pts = points.astype(np.int32).reshape((-1, 1, 2))

# Axis calibration
P0 = points[0]  # Origin
P1 = points[1]  # X-axis (610 mm)
P3 = points[3]  # Y-axis (400 mm)

# Define X and Y basis vectors
vec_x = P1 - P0
vec_y = P3 - P0
unit_x = vec_x / np.linalg.norm(vec_x)
unit_y = vec_y / np.linalg.norm(vec_y)

# Lengths of sides in pixels
len_x_pixels = np.linalg.norm(vec_x)
len_y_pixels = np.linalg.norm(vec_y)

# mm per pixel
scale_x = 610.0 / len_x_pixels
scale_y = 400.0 / len_y_pixels

# -------- Load YOLOv11 Model --------
MODEL_PATH = '/home/bestautomation/Downloads/best.pt'
model = YOLO(MODEL_PATH)

# -------- Setup OAK Camera --------
OAK_POE_IP = "169.254.1.222"
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# -------- Start Feed with Cartesian Projection --------
with dai.Device(pipeline, dai.DeviceInfo(OAK_POE_IP)) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        in_frame = q.get().getCvFrame()
        frame = in_frame.copy()

        # Draw calibrated polygon
        cv2.polylines(frame, [drawn_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Label origin
        cv2.putText(frame, "Origin (0,0)", tuple(P0.astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw axis lengths
        axis_labels = ["610 mm", "400 mm", "600 mm", "400 mm"]
        for i in range(4):
            mid = ((points[i] + points[(i + 1) % 4]) / 2).astype(int)
            cv2.putText(frame, axis_labels[i], tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # --- YOLO Prediction ---
        results = model(frame, imgsz=640, verbose=False)[0]
        queue = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Center of bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            center = np.array([cx, cy], dtype=np.float32)

            # Vector from origin to center
            vec = center - P0

            # Project onto x and y axes
            proj_x = np.dot(vec, unit_x)
            proj_y = np.dot(vec, unit_y)

            # mm coordinates
            mm_x = proj_x * scale_x
            mm_y = proj_y * scale_y

            # Intersection points in image
            point_on_x = P0 + proj_x * unit_x
            point_on_y = P0 + proj_y * unit_y

            ix, iy = point_on_x.astype(int)
            jx, jy = point_on_y.astype(int)

            # Draw center
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw projections as dashed lines (manual)
            for i in range(0, 100, 10):
                p1 = tuple(((cx - (cx - ix) * i / 100, cy - (cy - iy) * i / 100)))
                p2 = tuple(((cx - (cx - ix) * (i + 5) / 100, cy - (cy - iy) * (i + 5) / 100)))
                cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 200, 200), 1)

                p3 = tuple(((cx - (cx - jx) * i / 100, cy - (cy - jy) * i / 100)))
                p4 = tuple(((cx - (cx - jx) * (i + 5) / 100, cy - (cy - jy) * (i + 5) / 100)))
                cv2.line(frame, tuple(map(int, p3)), tuple(map(int, p4)), (200, 200, 0), 1)

            # Mark intersection points
            cv2.circle(frame, (ix, iy), 4, (0, 200, 200), -1)
            cv2.circle(frame, (jx, jy), 4, (200, 200, 0), -1)

            # Display labels
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

            coord_text = f"X={mm_x:.1f} mm, Y={mm_y:.1f} mm"
            print(mm_x)
            print(mm_y)
            cv2.putText(frame, coord_text, (cx + 5, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Show window
        cv2.imshow("YOLOv11 Cartesian Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
