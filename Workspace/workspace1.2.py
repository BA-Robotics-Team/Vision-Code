import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
from collections import deque
import math


class LiveCornerDetectionSystem:
    def __init__(self, corner_size_cm: float = 3.0, camera_id: int = 1):
        self.corner_size_cm = corner_size_cm
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False

        self.min_square_area = 500
        self.max_square_area = 10000

        self.corner_history = deque(maxlen=5)
        self.workspace_corners = []
        self.last_valid_corners = []

    def initialize_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        return True

    def create_black_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 45])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_squares(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        square_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_square_area <= area <= self.max_square_area):
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and self.is_square(approx, area):
                square_contours.append(approx)
                if len(square_contours) == 4:
                    break

        return square_contours

    def is_square(self, contour: np.ndarray, area: float) -> bool:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0
        bbox_area = w * h
        return abs(aspect_ratio - 1.0) < 0.4 and 0.6 < area / bbox_area < 1.2

    def get_inner_corners(self, square_contours: List[np.ndarray], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        if len(square_contours) != 4:
            return []

        image_center = (shape[1] // 2, shape[0] // 2)
        inner_corners = []

        for contour in square_contours:
            corners = contour.reshape(-1, 2)
            inner_corner = min(corners, key=lambda pt: math.hypot(pt[0] - image_center[0], pt[1] - image_center[1]))
            inner_corners.append(inner_corner)

        if len(inner_corners) == 4:
            sorted_corners = self.sort_cyclic(inner_corners)
            self.workspace_corners = sorted_corners
            return sorted_corners

        return []

    def sort_cyclic(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        center_x = sum(pt[0] for pt in corners) / 4
        center_y = sum(pt[1] for pt in corners) / 4
        sorted_pts = sorted(corners, key=lambda pt: math.atan2(pt[1] - center_y, pt[0] - center_x))
        return sorted_pts

    def draw_workspace_border(self, frame: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 3)
        return frame

    def run(self):
        if not self.initialize_camera():
            return

        self.is_running = True
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            mask = self.create_black_mask(frame)
            squares = self.detect_squares(mask)
            corners = self.get_inner_corners(squares, frame.shape[:2])
            frame = self.draw_workspace_border(frame, corners)

            cv2.imshow('Workspace Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    system = LiveCornerDetectionSystem()
    system.run()
