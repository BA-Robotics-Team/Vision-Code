import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import threading
from collections import deque
import math

class LiveCornerDetectionSystem:
    """
    Real-time corner detection system for live video streams
    """
    
    def __init__(self, corner_size_cm: float = 3.0, camera_id: int = 1 ):
        """
        Initialize the live corner detection system
        
        Args:
            corner_size_cm: Expected size of corner squares in centimeters
            camera_id: Camera device ID (0 for default camera)
        """
        self.corner_size_cm = corner_size_cm
        self.camera_id = camera_id
        
        # Video capture setup
        self.cap = None
        self.is_running = False
        
        # Detection parameters
        self.detection_confidence = 0.7
        self.min_square_area = 500
        self.max_square_area = 10000
        
        # Smoothing and tracking
        self.corner_history = deque(maxlen=5)  # Store last 5 detections for smoothing
        self.workspace_corners = []
        self.last_valid_corners = []
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Processing modes
        self.show_debug = True
        self.show_intermediate = False
        self.record_video = False
        self.video_writer = None
        
        # UI state
        self.paused = False
        self.calibration_mode = False
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture
        
        Returns:
            True if camera initialized successfully
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {width}x{height} @ {fps}fps")
        return True
    
    def create_optimized_color_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized color filter for real-time processing
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask highlighting black regions
        """
        # Convert to HSV (faster than multiple color space conversions)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Optimized black color range
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 45])  # Slightly tighter range for speed
        
        # Create mask
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Faster morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return black_mask
    
    def detect_squares_optimized(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Optimized square detection for real-time processing
        
        Args:
            mask: Binary mask image
            
        Returns:
            List of square contours
        """
        # Find contours with faster approximation
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        square_contours = []
        
        for contour in contours:
            # Quick area filter
            area = cv2.contourArea(contour)
            if area < self.min_square_area or area > self.max_square_area:
                continue
            
            # Quick perimeter check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Approximate to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check for 4 vertices and square shape
            if len(approx) == 4 and self._is_square_shaped_fast(approx, area):
                square_contours.append(approx)
                
                # Limit to 4 squares for performance
                if len(square_contours) >= 4:
                    break
        
        return square_contours
    
    def _is_square_shaped_fast(self, contour: np.ndarray, area: float) -> bool:
        """
        Fast square shape validation
        
        Args:
            contour: 4-vertex contour
            area: Pre-calculated area
            
        Returns:
            True if the contour is square-shaped
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Quick aspect ratio check
        aspect_ratio = float(w) / h if h > 0 else 0
        if abs(aspect_ratio - 1.0) > 0.4:
            return False
        
        # Check if contour area is reasonable compared to bounding box
        bbox_area = w * h
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            return 0.6 < fill_ratio < 1.2
        
        return False
    
    def get_inner_corners_stable(self, square_contours: List[np.ndarray], 
                                image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Extract inner corners with stability tracking
        
        Args:
            square_contours: List of square contours
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of stable inner corner points
        """
        if len(square_contours) == 0:
            return self.last_valid_corners if self.last_valid_corners else []
        
        inner_corners = []
        image_center = (image_shape[1] // 2, image_shape[0] // 2)
        
        for contour in square_contours:
            corners = contour.reshape(-1, 2)
            
            # Find the corner closest to image center (inner corner)
            min_dist = float('inf')
            inner_corner = tuple(corners[0])
            
            for corner in corners:
                dist = math.sqrt((corner[0] - image_center[0])**2 + 
                               (corner[1] - image_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    inner_corner = tuple(corner)
            
            inner_corners.append(inner_corner)
        
        # Store for smoothing
        if len(inner_corners) == 4:
            self.corner_history.append(inner_corners)
            self.last_valid_corners = inner_corners
        
        return self._smooth_corners(inner_corners)
    
    def _smooth_corners(self, current_corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Apply temporal smoothing to corner positions
        
        Args:
            current_corners: Current frame corner positions
            
        Returns:
            Smoothed corner positions
        """
        if len(self.corner_history) < 2:
            return current_corners
        
        # Average last few detections
        smoothed_corners = []
        history_array = np.array(list(self.corner_history))
        
        for i in range(len(current_corners)):
            if i < history_array.shape[1]:
                avg_x = int(np.mean(history_array[:, i, 0]))
                avg_y = int(np.mean(history_array[:, i, 1]))
                smoothed_corners.append((avg_x, avg_y))
            else:
                smoothed_corners.append(current_corners[i])
        
        return smoothed_corners
    
    def create_workspace_overlay_live(self, image: np.ndarray, 
                                    inner_corners: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create real-time workspace overlay
        
        Args:
            image: Original image
            inner_corners: List of inner corner points
            
        Returns:
            Image with workspace overlay
        """
        result_image = image.copy()
        
        if len(inner_corners) >= 4:
            # Sort corners for proper rectangle
            corners_array = np.array(inner_corners[:4], dtype=np.int32)
            
            # Draw workspace boundary
            cv2.polylines(result_image, [corners_array], True, (0, 255, 0), 2)
            
            # Semi-transparent fill
            overlay = result_image.copy()
            cv2.fillPoly(overlay, [corners_array], (0, 255, 0))
            result_image = cv2.addWeighted(result_image, 0.85, overlay, 0.15, 0)
            
            # Mark corners
            for i, corner in enumerate(corners_array):
                cv2.circle(result_image, corner, 6, (0, 0, 255), -1)
                cv2.putText(result_image, f'{i+1}', 
                          (corner[0] + 8, corner[1] - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self.workspace_corners = inner_corners[:4]
        
        elif len(self.last_valid_corners) >= 4:
            # Use last valid corners if current detection failed
            corners_array = np.array(self.last_valid_corners[:4], dtype=np.int32)
            cv2.polylines(result_image, [corners_array], True, (0, 255, 255), 2)
        
        return result_image
    
    def add_ui_overlay(self, image: np.ndarray, squares_detected: int) -> np.ndarray:
        """
        Add UI information overlay to the image
        
        Args:
            image: Input image
            squares_detected: Number of squares detected
            
        Returns:
            Image with UI overlay
        """
        overlay_image = image.copy()
        
        # Background for text
        cv2.rectangle(overlay_image, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay_image, (10, 10), (300, 120), (255, 255, 255), 1)
        
        # Status information
        status_color = (0, 255, 0) if squares_detected == 4 else (0, 255, 255)
        
        cv2.putText(overlay_image, f"FPS: {self.current_fps:.1f}", 
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(overlay_image, f"Squares: {squares_detected}/4", 
                   (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        cv2.putText(overlay_image, f"Workspace: {'ACTIVE' if len(self.workspace_corners) == 4 else 'INACTIVE'}", 
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # Controls
        cv2.putText(overlay_image, "Controls: 'q'-quit, 'p'-pause, 'd'-debug", 
                   (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(overlay_image, "'r'-record, 'c'-calibrate", 
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay_image
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def setup_video_recording(self, output_path: str = "workspace_detection.avi"):
        """
        Setup video recording
        
        Args:
            output_path: Path for output video file
        """
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20  # Recording FPS
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Video recording started: {output_path}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detection_info)
        """
        # Step 1: Color filtering
        black_mask = self.create_optimized_color_filter(frame)
        
        # Step 2: Square detection
        square_contours = self.detect_squares_optimized(black_mask)
        
        # Step 3: Extract inner corners
        inner_corners = self.get_inner_corners_stable(square_contours, frame.shape[:2])
        
        # Step 4: Create workspace overlay
        result_frame = self.create_workspace_overlay_live(frame, inner_corners)
        
        # Step 5: Add UI overlay
        result_frame = self.add_ui_overlay(result_frame, len(square_contours))
        
        # Prepare detection info
        detection_info = {
            'squares_detected': len(square_contours),
            'inner_corners': inner_corners,
            'workspace_active': len(self.workspace_corners) == 4,
            'black_mask': black_mask if self.show_intermediate else None
        }
        
        return result_frame, detection_info
    
    def run_live_detection(self):
        """
        Main loop for live corner detection
        """
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("Live corner detection started. Press 'q' to quit.")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  'd' - Toggle debug view")
        print("  'r' - Start/Stop recording")
        print("  'c' - Calibration mode")
        print("  's' - Save current workspace")
        
        try:
            while self.is_running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break
                    
                    # Process frame
                    processed_frame, detection_info = self.process_frame(frame)
                    
                    # Show intermediate results if requested
                    if self.show_intermediate and detection_info['black_mask'] is not None:
                        cv2.imshow('Black Mask', detection_info['black_mask'])
                    
                    # Display main result
                    cv2.imshow('Live Corner Detection', processed_frame)
                    
                    # Record video if enabled
                    if self.record_video and self.video_writer is not None:
                        self.video_writer.write(processed_frame)
                    
                    # Update FPS
                    self.update_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif key == ord('d'):
                    self.show_intermediate = not self.show_intermediate
                    print(f"Debug view: {'ON' if self.show_intermediate else 'OFF'}")
                elif key == ord('r'):
                    if not self.record_video:
                        self.setup_video_recording()
                        self.record_video = False
                    else:
                        self.record_video = False
                        if self.video_writer:
                            self.video_writer.release()
                            #self.video_writer = None
                        print("Recording stopped")
                elif key == ord('c'):
                    self.calibration_mode = not self.calibration_mode
                    print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
                elif key == ord('s'):
                    self.save_current_workspace()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def save_current_workspace(self):
        """Save current workspace coordinates"""
        if len(self.workspace_corners) == 4:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"workspace_coords_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Workspace coordinates saved at {time.ctime()}\n")
                f.write("Corner coordinates (x, y):\n")
                for i, corner in enumerate(self.workspace_corners):
                    f.write(f"Corner {i+1}: ({corner[0]}, {corner[1]})\n")
            
            print(f"Workspace coordinates saved to {filename}")
        else:
            print("No valid workspace detected to save")
    
    def get_current_workspace(self) -> List[Tuple[int, int]]:
        """
        Get current workspace coordinates for robot integration
        
        Returns:
            List of current workspace corner coordinates
        """
        return self.workspace_corners.copy()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")


class RobotWorkspaceInterface:
    """
    Interface for robot integration with live workspace detection
    """
    
    def __init__(self, detection_system: LiveCornerDetectionSystem):
        self.detection_system = detection_system
        self.workspace_callback = None
        self.monitoring_thread = None
        self.monitoring_active = False
    
    def set_workspace_callback(self, callback_function):
        """
        Set callback function to be called when workspace changes
        
        Args:
            callback_function: Function to call with workspace coordinates
        """
        self.workspace_callback = callback_function
    
    def start_workspace_monitoring(self):
        """Start monitoring workspace changes in separate thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_workspace)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            print("Workspace monitoring started")
    
    def stop_workspace_monitoring(self):
        """Stop workspace monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Workspace monitoring stopped")
    
    def _monitor_workspace(self):
        """Monitor workspace changes and trigger callbacks"""
        last_workspace = []
        
        while self.monitoring_active:
            current_workspace = self.detection_system.get_current_workspace()
            
            # Check if workspace changed significantly
            if self._workspace_changed(last_workspace, current_workspace):
                if self.workspace_callback and len(current_workspace) == 4:
                    self.workspace_callback(current_workspace)
                last_workspace = current_workspace.copy()
            
            time.sleep(0.1)  # Check 10 times per second
    
    def _workspace_changed(self, old_workspace: List[Tuple[int, int]], 
                          new_workspace: List[Tuple[int, int]], 
                          threshold: int = 10) -> bool:
        """
        Check if workspace changed significantly
        
        Args:
            old_workspace: Previous workspace coordinates
            new_workspace: New workspace coordinates
            threshold: Minimum change threshold in pixels
            
        Returns:
            True if workspace changed significantly
        """
        if len(old_workspace) != len(new_workspace):
            return True
        
        if len(new_workspace) != 4:
            return False
        
        for old_pt, new_pt in zip(old_workspace, new_workspace):
            distance = math.sqrt((old_pt[0] - new_pt[0])**2 + (old_pt[1] - new_pt[1])**2)
            if distance > threshold:
                return True
        
        return False
    
    def get_workspace_info(self) -> Dict:
        """
        Get comprehensive workspace information
        
        Returns:
            Dictionary with workspace information
        """
        workspace = self.detection_system.get_current_workspace()
        
        if len(workspace) == 4:
            # Calculate workspace dimensions
            width = max(abs(workspace[1][0] - workspace[0][0]), 
                       abs(workspace[3][0] - workspace[2][0]))
            height = max(abs(workspace[2][1] - workspace[0][1]), 
                        abs(workspace[3][1] - workspace[1][1]))
            
            return {
                'active': True,
                'corners': workspace,
                'width_pixels': width,
                'height_pixels': height,
                'center': ((workspace[0][0] + workspace[2][0]) // 2,
                          (workspace[0][1] + workspace[2][1]) // 2)
            }
        else:
            return {
                'active': False,
                'corners': [],
                'width_pixels': 0,
                'height_pixels': 0,
                'center': (0, 0)
            }


# Example usage and main function
def main():
    """
    Main function for live corner detection
    """
    print("Starting Live Corner Detection System")
    print("="*50)
    
    # Initialize the detection system
    detector = LiveCornerDetectionSystem(corner_size_cm=3.0, camera_id=1)
    
    # Set up robot interface (optional)
    robot_interface = RobotWorkspaceInterface(detector)
    
    # Example callback function for robot integration
    def workspace_updated(corners):
        print(f"Robot: New workspace detected - {corners}")
        # Here you would send coordinates to your robot controller
    
    # robot_interface.set_workspace_callback(workspace_updated)
    # robot_interface.start_workspace_monitoring()
    
    try:
        # Start live detection
        detector.run_live_detection()
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # robot_interface.stop_workspace_monitoring()
        print("System shutdown complete")


if __name__ == "__main__":
    main()

