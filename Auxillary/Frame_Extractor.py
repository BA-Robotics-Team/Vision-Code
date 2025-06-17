import cv2
import os

# === USER INPUT ===
video_path = ''     # Path to your video file
output_folder = ''
interval_sec = 0.75                # Time gap between frames to extract

# === SETUP ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * interval_sec)
frame_count = 0
saved_count = 0

os.makedirs(output_folder, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"image_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to '{output_folder}'")
