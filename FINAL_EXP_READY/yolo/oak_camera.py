import depthai as dai
import cv2
import time
import os

# Create DepthAI pipeline
pipeline = dai.Pipeline()

# Create a ColorCamera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Create output stream node
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Create directory to save captured images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print("✅ OAK-1 camera connected. Starting video feed. Press 'q' to quit, 'c' to capture.")

    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    frame_count = 0

    while True:
        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()
        frame = cv2.flip(frame,1)
        # Show the frame
        cv2.imshow("OAK-1 Live Feed", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("🔴 Quitting...")
            break
        elif key == ord('c'):
            # Save the current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Image saved: {filename}")

    cv2.destroyAllWindows()
