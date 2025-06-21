from ultralytics import YOLO
import math as m
import cv2
import time


model = YOLO("Models\\Card_Box\\2\\runs\\train\\yolov8_obb_biscuit\\weights\\best.pt")


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    img=cv2.flip(img,-1)

    # Run inference
    results = model.predict(img)

    roll_deg = None
    for result in results:
        for obb in result.obb.xywhr:
            roll_rad = obb[4].item()
            x,y=obb[0].item(),obb[1].item()
            roll_deg = m.degrees(roll_rad)

    # Plot and show results
    cv2.imshow("Result", results[0].plot())

    # Save roll angle if detected
    if roll_deg is not None:
        with open("file.txt", 'a') as f:
            f.write(f"{roll_deg:.2f}\n")

    # Break on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
