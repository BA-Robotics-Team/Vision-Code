import cv2
import numpy as np

camera_index = 1
zoom_out_factor = 1 # Try 0.5 for half-size (zoomed out)

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Cannot open USB camera.")
    exit()

print("✅ USB Camera feed started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize (scale down)
    small_frame = cv2.resize(frame, None, fx=zoom_out_factor, fy=zoom_out_factor)

    # Create a black canvas the same size as original frame
    canvas = np.zeros_like(frame)

    # Get coordinates to center the small frame in canvas
    h, w = small_frame.shape[:2]
    H, W = canvas.shape[:2]
    x = (W - w) // 2
    y = (H - h) // 2

    # Paste the small frame in the middle of the canvas
    canvas[y:y+h, x:x+w] = small_frame

    cv2.imshow("Zoomed-Out USB Camera Feed", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Exiting camera feed.")
        break

cap.release()
cv2.destroyAllWindows()
