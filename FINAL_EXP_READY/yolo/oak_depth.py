import depthai as dai
import cv2

OAK_POE_IP = "169.254.1.222"# Replace with actual IP from find_oak_poe.py

pipeline = dai.Pipeline()

# Setup camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)

# Output stream
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Connect to the camera
with dai.Device(pipeline, dai.DeviceInfo(OAK_POE_IP)) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    while True:
        frame = q.get().getCvFrame()
        cv2.imshow("OAK PoE Live Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
