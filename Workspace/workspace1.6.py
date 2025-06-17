import cv2
import numpy as np
from ultralytics import YOLO
# Constants
WORKSPACE_WIDTH = 11.6  # cm
WORKSPACE_HEIGHT = 17   # cm

def get_black_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))

def detect_squares(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 300:
            squares.append(approx)
        if len(squares) == 4: break
    return squares

def get_inner_corners(squares, img_center):
    corners = []
    for sq in squares:
        pts = sq.reshape(4, 2)
        inner = min(pts, key=lambda p: np.linalg.norm(p - img_center))
        corners.append(tuple(inner))
    if len(corners) == 4:
        sorted_pts = sorted(corners, key=lambda p: (p[1], p[0]))
        bottom = sorted(sorted_pts[2:], key=lambda p: p[0])
        top = sorted(sorted_pts[:2], key=lambda p: p[0])
        return [bottom[0], bottom[1], top[1], top[0]]  # bl, br, tr, tl
    return []

def overlay_border(img, pts):
    if len(pts) != 4: return img
    bl, br, tr, tl = pts
    px_per_cm_x = (br[0] - bl[0]) / WORKSPACE_WIDTH
    px_per_cm_y = (bl[1] - tl[1]) / WORKSPACE_HEIGHT
    origin = bl
    for i, pt in enumerate(pts):
        coord = ((pt[0] - origin[0]) / px_per_cm_x, (origin[1] - pt[1]) / px_per_cm_y)
        cv2.circle(img, pt, 5, (0,255,0), -1)
        cv2.putText(img, f"({coord[0]:.1f},{coord[1]:.1f})", (pt[0]+5, pt[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.line(img, pt, pts[(i+1)%4], (255,0,0), 2)

    return img

def main():
    cap = cv2.VideoCapture(1)
    model=YOLO("")#will put this later ignore for now
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, -1)
        frame=model.predict(frame)[0].plot()
        mask = get_black_mask(frame)
        squares = detect_squares(mask)
        corners = get_inner_corners(squares, np.array(frame.shape[1::-1]) // 2)
        out = overlay_border(frame, corners)
        cv2.imshow("Workspace", out)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
