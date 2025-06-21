import cv2
import numpy as np
from ultralytics import YOLO

# Load model
Y = YOLO("runs\\train\\yolov8_obb_biscuit\\weights\\best.pt")

# Run prediction
result = Y.predict("50NEW\\Valid\\Images\\frame_00476.jpg", save=False, show=False, verbose=False)
img_path = "50NEW\\Valid\\Images\\frame_00476.jpg"

# Read the original image
image = cv2.imread(img_path)

# Font for writing text
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop through detections
for obb in result[0].obb.xyxyxyxy:
    points = obb.cpu().numpy().reshape(-1, 2).astype(int)
    
    # Draw box edges and coordinates
    for i in range(4):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % 4])
        
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(image, pt1, 4, (0, 0, 255), -1)
        
        # Write coordinates near each point
        text = f"{pt1[0]}, {pt1[1]}"
        cv2.putText(image, text, (pt1[0] + 5, pt1[1] - 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

# Show and save
cv2.imshow("OBB Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("obb_output_with_coords.jpg", image)

# Print normalized values (optional)
print("Normalized OBB:", result[0].obb.xyxyxyxyn)
