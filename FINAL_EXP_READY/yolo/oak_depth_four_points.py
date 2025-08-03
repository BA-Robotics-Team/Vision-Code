import depthai as dai
import cv2
import json
import numpy as np

# ---------- Load and parse COCO JSON ----------
with open("/home/bestautomation/Downloads/labels_my-project-name_2025-07-22-01-54-08.json", "r") as f:
    data = json.load(f)

# Get the segmentation points
segmentation = data["annotations"][0]["segmentation"][0]

# Convert flat list to Nx2 array of (x, y) points
roi_points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))

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

# ---------- Start camera and draw ROI ----------
with dai.Device(pipeline, dai.DeviceInfo(OAK_POE_IP)) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        frame = q.get().getCvFrame()

        # Draw polygonal ROI from segmentation
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("OAK PoE Live Feed with ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
