import cv2
from picamera2 import Picamera2

# Initialize PiCamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 720)})
picam2.configure(preview_config)
picam2.start()

print("Starting live feed. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    cv2.imshow("Live Feed - RasPi Cam v3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
