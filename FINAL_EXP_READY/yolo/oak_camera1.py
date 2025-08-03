import depthai as dai
import cv2
import numpy as np

# Create DepthAI pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    print("✅ OAK-1 camera connected. Press 'q' to quit.")

    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        frame = video_queue.get().getCvFrame()

        # 1️⃣ Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2️⃣ Black Color Mask
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        mask_black = cv2.inRange(frame, lower_black, upper_black)
        black_masked = cv2.bitwise_and(frame, frame, mask=mask_black)

        # 3️⃣ Edge Detection
        edges = cv2.Canny(gray, 100, 200)

        # 4️⃣ Gaussian Blur
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)

        # 5️⃣ Thresholding (binary)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Show all outputs
        cv2.imshow("Original", frame)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Black Color Mask", black_masked)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("Gaussian Blur", blurred)
        cv2.imshow("Binary Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 Exiting...")
            break

    cv2.destroyAllWindows()
