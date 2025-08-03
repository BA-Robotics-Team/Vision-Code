import depthai as dai
import cv2
import json
import numpy as np

# ---------- Load and parse COCO JSON ----------
with open("/home/bestautomation/Downloads/labels_my-project-name_2025-07-22-01-54-08.json", "r") as f:
    data = json.load(f)

# Get polygon points and reshape
segmentation = data["annotations"][0]["segmentation"][0]
points = np.array(segmentation, dtype=np.float32).reshape((-1, 2))  # shape (4,2)

# Known real-world distances in mm (ordered as sides between the 4 points)
real_world_lengths_mm = [610, 400, 600, 400]  # Clockwise

# Compute pixel distances between points
pixel_lengths = [np.linalg.norm(points[i] - points[(i+1)%4]) for i in range(4)]

# Compute pixel-to-mm scale for each side
pixel_to_mm = [real / pix for real, pix in zip(real_world_lengths_mm, pixel_lengths)]

# ---------- Setup OAK-D PoE camera pipeline ----------
OAK_POE_IP = "169.254.1.222"  # Replace with actual IP

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Reshape for drawing
drawn_pts = points.astype(np.int32).reshape((-1, 1, 2))

# ---------- Start camera feed and annotate distances ----------
with dai.Device(pipeline, dai.DeviceInfo(OAK_POE_IP)) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        frame = q.get().getCvFrame()

        # Draw polygon
        cv2.polylines(frame, [drawn_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw distances
        for i in range(4):
            pt1 = points[i]
            pt2 = points[(i+1)%4]
            mid = ((pt1 + pt2) / 2).astype(int)
            length_mm = real_world_lengths_mm[i]
            label = f"{length_mm} mm"

            # Draw label
            cv2.putText(frame, label, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw origin label
        origin = tuple(points[0].astype(int))
        cv2.putText(frame, "Origin (0,0)", origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("OAK PoE Calibrated ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
