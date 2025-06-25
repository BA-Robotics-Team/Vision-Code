#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import numpy as np
import math
from collections import deque


def main():
    MODEL_PATH = r"E:\BA_DOBOT\Vision-Code\Models\Card_Box\1\runs\BEx\weights\best.pt"
    
    # Initialize
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(1)
    angle_history = deque(maxlen=5)
    
    if not cap.isOpened():
        print("Camera not found")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, -1)
            results = model.predict(frame, verbose=False)
            
            for result in results:
                if result.obb is not None and len(result.obb.xywhr) > 0:
                    obb = result.obb.xywhr[0].cpu().numpy()
                    
                    # Calculate angle
                    rotation, width, height = obb[4], obb[2], obb[3]
                    angle = rotation if width >= height else rotation + math.pi/2
                    angle = math.degrees(angle) % 360
                    
                    # Smooth angle
                    angle_history.append(angle)
                    if len(angle_history) > 1:
                        cos_sum = sum(math.cos(math.radians(a)) for a in angle_history)
                        sin_sum = sum(math.sin(math.radians(a)) for a in angle_history)
                        angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
                    
                    # Draw visualization
                    cx, cy = int(obb[0]), int(obb[1])
                    w, h, rot = obb[2], obb[3], obb[4]
                    
                    # Object bounding box
                    corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(rot)))
                    cv2.drawContours(frame, [np.int0(corners)], -1, (0, 255, 0), 2)
                    
                    # Reference line (horizontal)
                    line_len = int(max(w, h) * 0.8)
                    cv2.line(frame, (cx - line_len//2, cy), (cx + line_len//2, cy), 
                            (100, 100, 100), 1)
                    
                    # Object orientation line
                    obj_angle = rotation if width >= height else rotation + math.pi/2
                    end_x = cx + int(line_len//2 * math.cos(obj_angle))
                    end_y = cy + int(line_len//2 * math.sin(obj_angle))
                    start_x = cx - int(line_len//2 * math.cos(obj_angle))
                    start_y = cy - int(line_len//2 * math.sin(obj_angle))
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 165, 255), 2)
                    
                    # Center point
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                    
                    # Clean angle display
                    cv2.putText(frame, f"{angle:.1f}°", (cx + 20, cy - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    break
            
            cv2.imshow("Rotation Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()