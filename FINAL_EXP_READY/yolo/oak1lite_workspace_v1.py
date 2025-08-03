import json
import cv2
import numpy as np
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional
import depthai as dai
from ultralytics import YOLO
import serial

# import serial  # Uncomment when STM32 is connected

class PickAndPlaceRobot:
    """Simplified Pick-and-Place Robot Controller optimized for RPi 5"""
    
    def __init__(self,serial_conn):
        # Configuration 
        self.coco_json_path = r'/home/bestautomation/yolo/coordinates.txt'
        self.yolo_model_path = r'/home/bestautomation/Downloads/best.pt'
        self.workspace_width_mm = 610
        self.workspace_height_mm = 400
        self.distance_threshold_mm = 10.0
        self.no_object_timeout = 30.0
        self.post_ack_video_duration = 5.0  # 3 seconds video feed after ACK
        
        # RPi 5 optimized settings
        self.camera_fps = 15  # Reduced FPS for RPi 5
        self.camera_resolution = (640, 480)  # Lower resolution for better performance
        
        # Initialize components
        self.roi_pixel_pts = None
        self.homography_matrix = None
        self.yolo_model = None
        self.camera_pipeline = None
        self.device = None
        self.video_queue = None
        self.processed_coords = []
        
        print("[INFO] Initializing Pick-and-Place Robot for RPi 5...")
        # CWSTM
        self._load_workspace_calibration()
        self._load_yolo_model()
        self._setup_camera()
        self.coorrec = 0
        self.serial_conn = serial_conn
        self.motion_completed = 0
        
        #print("[INFO] Initialising Communication")
        #self.ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
        #print("Serial port opened")
    
    def _load_workspace_calibration(self):
        """Load ROI from COCO JSON and compute homography"""
        print("[INFO] Loading workspace calibration...")
        
        with open(self.coco_json_path, "r") as f:
            coco_data = json.load(f)
        
        # Extract ROI polygon points
        segmentation = coco_data["annotations"][0]["segmentation"][0]
        self.roi_pixel_pts = [(segmentation[i], segmentation[i + 1]) 
                             for i in range(0, len(segmentation), 2)]
        
        # Define real-world coordinates (mm) - origin at top-left
        real_world_pts = np.array([
            [0, 0],                                    # top-left (origin)
            [self.workspace_width_mm, 0],             # top-right
            [self.workspace_width_mm, self.workspace_height_mm],  # bottom-right
            [0, self.workspace_height_mm]             # bottom-left
        ], dtype=np.float32)
        
        # Convert pixel points to numpy array
        pixel_pts = np.array(self.roi_pixel_pts, dtype=np.float32)
        
        # Compute homography matrix
        self.homography_matrix, _ = cv2.findHomography(pixel_pts, real_world_pts)
        
        print(f"[INFO] Workspace calibration complete: {self.workspace_width_mm}mm x {self.workspace_height_mm}mm")
    
    def _load_yolo_model(self):
        """Load YOLO model using Ultralytics"""
        print("[INFO] Loading YOLO model...")
        try:
            # For RPi 5 - consider using a smaller model for better performance
            self.yolo_model = YOLO(self.yolo_model_path)
            # Set device to CPU for RPi 5
            self.yolo_model.to('cpu')
            print("[INFO] YOLO model loaded successfully on CPU")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            raise   
    
    def _setup_camera(self):
        """Setup DepthAI camera optimized for RPi 5"""
        print("[INFO] Setting up camera for RPi 5...")
        
        self.camera_pipeline = dai.Pipeline()
        
        # Create color camera with RPi 5 optimized settings
        cam = self.camera_pipeline.createColorCamera()
        cam.setPreviewSize(*self.camera_resolution)  # Lower resolution for RPi 5
        cam.setFps(self.camera_fps)  # Reduced FPS for RPi 5
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # Create output
        xout_video = self.camera_pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        cam.preview.link(xout_video.input)  # Use preview instead of video for lower resolution
        
        print(f"[INFO] Camera setup complete - {self.camera_resolution[0]}x{self.camera_resolution[1]} @ {self.camera_fps}fps")
    
    def start_camera(self):
        """Start camera device"""
        self.device = dai.Device(self.camera_pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
        print("[INFO] Camera started")
    
    def stop_camera(self):
        """Stop camera"""
        if self.device:
            self.device.close()
            print("[INFO] Camera stopped")
    
    def capture_frame(self):
        """Capture frame from camera"""
        if self.video_queue is None:
            return None
        
        try:
            in_frame = self.video_queue.get()
            return in_frame.getCvFrame()
        except:
            return None
    
    def pixel_to_real_world(self, pixel_point):
        """Convert pixel coordinates to real-world mm coordinates"""
        input_pt = np.array([[pixel_point]], dtype=np.float32)
        real_world_pt = cv2.perspectiveTransform(input_pt, self.homography_matrix)[0][0]
        return float(real_world_pt[0]), float(real_world_pt[1])
    
    def is_point_in_roi(self, pixel_point):
        """Check if point is inside ROI"""
        roi_contour = np.array(self.roi_pixel_pts, dtype=np.int32)
        result = cv2.pointPolygonTest(roi_contour, pixel_point, False)
        return result >= 0
    
    def detect_objects(self, frame):
        """Detect objects using YOLO - optimized for RPi 5"""
        try:
            # For RPi 5 performance, consider reducing image size for inference
            inference_frame = cv2.resize(frame, (320, 240))  # Smaller size for faster inference
            results = self.yolo_model(inference_frame, verbose=False)
            detections = []
            
            # Scale factor to convert back to original frame size
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates and scale back
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > 0.5:  # Confidence threshold
                            detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
            
            return detections
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return []
    
    def get_bbox_center(self, bbox):
        """Get center of bounding box"""
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_duplicate(self, new_coord):
        """Check if coordinate is too close to processed ones"""
        for processed_coord in self.processed_coords:
            if self.calculate_distance(new_coord, processed_coord) < self.distance_threshold_mm:
                return True
        return False
    
    def send_coordinate(self, x, y):
        """Send coordinate to STM32 (simulated for now)"""
        coordinate_str = f"$COORX{x:.2f}Y{y:.2f}theta000.00END#"
        print(f"[SEND] {coordinate_str}")
        try:
            while True:
                self.serial_conn.write((coordinate_str + "\r\n").encode())
                #print("[INFO] Coordinate sent. Waiting for STM32 acknowledgment...")

                response = self.serial_conn.readline().decode(errors='ignore').strip()
                #print(f"[RECV] {response}")
            
                if response == "COORRECEIVED":
                    print("[INFO] Acknowledgment received from STM32.")
                    #return True
                    while self.motion_completed == 0:
                        response = self.serial_conn.readline().decode(errors='ignore').strip()
                        print(f"[RECV] {response}")
                        if response == "MOTIONCOMPLETED":
                            #self.motion_completed =1
                            #print("[INFO] Acknowledgment received from STM32.")
                            return True
                        else:
                            #print("[WARN] Unexpected response. Resending...")
                            time.sleep(0.2)
                    
                else:
                    #print("[WARN] Unexpected response. Resending...")
                    time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] Failed to send coordinates: {e}")
            return False
        # For STM32 connection, uncomment below:
        # if self.serial_conn:
        #     try:
        #         self.serial_conn.write(f"{coordinate_str}\n".encode())
        #         response = self.serial_conn.readline().decode().strip()
        #         return response == "ACK"
        #     except Exception as e:
        #         print(f"[ERROR] Serial communication failed: {e}")
        #         return False
        
        # Simulate waiting for ACK
        #v=0
        #while(v==0):
            #user_input = input("Enter '1' to simulate ACK: ")
            #if (user_input.strip() == "1"):
                #v=1
                #return user_input.strip() == "1"
            
            
    
    def run_post_ack_video_feed(self):
        """Run video feed for specified duration after receiving ACK"""
        print(f"[INFO] Running post-ACK video feed for {self.post_ack_video_duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < self.post_ack_video_duration:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Create visualization with status
            vis_frame = frame.copy()
            vis_frame = self.draw_workspace(vis_frame)
            
            # Show remaining time
            remaining_time = self.post_ack_video_duration - (time.time() - start_time)
            cv2.putText(vis_frame, f"Post-ACK Video Feed - {remaining_time:.1f}s remaining", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, "Waiting for robot to complete action...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Optional: Still detect and show objects during this period
            detections = self.detect_objects(frame)
            valid_coords, valid_detections = self.filter_detections_by_roi(detections)
            vis_frame = self.draw_detections(vis_frame, valid_detections, valid_coords)
            
            cv2.imshow("Pick and Place System", vis_frame)
            
            # Allow early exit
            if cv2.waitKey(33) & 0xFF == ord('q'):  # 33ms for ~30fps display
                return False
        
        print("[INFO] Post-ACK video feed completed")
        return True
    
    def filter_detections_by_roi(self, detections):
        """Filter detections by ROI and convert to real-world coordinates"""
        valid_coords = []
        valid_detections = []
        
        for detection in detections:
            center_pixel = self.get_bbox_center(detection)
            
            # Check if in ROI
            if self.is_point_in_roi(center_pixel):
                real_coord = self.pixel_to_real_world(center_pixel)
                
                # Check if not duplicate
                if not self.is_duplicate(real_coord):
                    valid_coords.append(real_coord)
                    valid_detections.append(detection)
        
        return valid_coords, valid_detections
    
    def draw_workspace(self, frame):
        """Draw ROI and detections on frame"""
        # Draw ROI polygon
        if self.roi_pixel_pts:
            pts = np.array(self.roi_pixel_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, "Workspace ROI", 
                       (int(self.roi_pixel_pts[0][0]), int(self.roi_pixel_pts[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame
    
    def draw_detections(self, frame, detections, real_coords=None):
        """Draw detection boxes and coordinates"""
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = detection
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Add labels
            label = f"Conf: {conf:.2f}"
            if real_coords and i < len(real_coords):
                real_x, real_y = real_coords[i]
                label += f" | ({real_x:.1f}, {real_y:.1f})mm"
            
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return frame
    
    def draw_target(self, frame, target_coord):
        """Draw target coordinate"""
        if target_coord:
            # Convert back to pixel for visualization
            real_world_pt = np.array([[target_coord]], dtype=np.float32)
            pixel_pt = cv2.perspectiveTransform(real_world_pt, 
                                              np.linalg.inv(self.homography_matrix))[0][0]
            
            pixel_x, pixel_y = int(pixel_pt[0]), int(pixel_pt[1])
            cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 255), 2)
            cv2.putText(frame, "TARGET", (pixel_x - 25, pixel_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame
    
    def process_frame(self, frame):
        """Process frame and return valid coordinates"""
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Filter by ROI and convert to real-world coordinates
        return self.filter_detections_by_roi(detections)
    
    def wait_for_no_objects(self):
        """Wait for no objects for specified timeout"""
        start_time = time.time()
        
        while time.time() - start_time < self.no_object_timeout:
            frame = self.capture_frame()
            if frame is None:
                continue
            
            coords, detections = self.process_frame(frame)
            
            if coords:  # Objects still found
                return False 
            
            # Show status
            vis_frame = frame.copy()
            vis_frame = self.draw_workspace(vis_frame)
            remaining_time = self.no_object_timeout - (time.time() - start_time)
            cv2.putText(vis_frame, f"No objects - waiting {remaining_time:.1f}s", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Pick and Place System", vis_frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                return True
                
        while True:
            final_msg  = "$ALLOBJECTSFINISHED#"
            self.serial_conn.write((final_msg + "\r\n").encode())
            time.sleep(0.2)
            response = self.serial_conn.readline().decode(errors='ignore').strip()
            if response == "Adieu":
                break
        
        return True  # Timeout reached
    
    def run(self):
        """Main control loop"""
        print("[INFO] Starting Pick-and-Place System on RPi 5")
        print("[INFO] Press 'q' to quit")
        
        try:
            self.start_camera()
            
            while True:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                valid_coords, valid_detections = self.process_frame(frame)
                
                # Create visualization
                vis_frame = frame.copy()
                vis_frame = self.draw_workspace(vis_frame)
                
                if not valid_coords:
                    # No valid objects
                    vis_frame = self.draw_detections(vis_frame, [])
                    cv2.putText(vis_frame, "No valid objects detected", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Pick and Place System", vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Wait for no objects
                    if self.wait_for_no_objects():
                        print("[INFO] No objects found for 30 seconds. Shutting down.")
                        break
                    continue
                
                # Draw all detections
                vis_frame = self.draw_detections(vis_frame, valid_detections, valid_coords)
                
                # Select leftmost coordinate
                target_coord = min(valid_coords, key=lambda coord: coord[0])
                
                # Draw target
                vis_frame = self.draw_target(vis_frame, target_coord)
                
                # Show frame
                cv2.imshow("Pick and Place System", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                print(f"[INFO] Target coordinate: ({target_coord[0]:.2f}, {target_coord[1]:.2f}) mm")
                
                # Send coordinate and wait for ACK
                if self.send_coordinate(target_coord[0], target_coord[1]):
                    print("[INFO] Coordinate sent successfully! ACK received.")
                    self.processed_coords.append(target_coord)
                    
                    # NEW: Run post-ACK video feed for 2-3 seconds
                    if not self.run_post_ack_video_feed():
                        break  # User pressed 'q' during video feed
                    
                    print("[INFO] Ready for next object detection...")
                else:
                    print("[WARNING] Failed to send coordinate or no ACK received")
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] System interrupted by user")
        except Exception as e:
            print(f"[ERROR] System error: {e}")
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
            print("[INFO] System shutdown complete")

def main():
    """Main function"""
    #initialise the flags : Refer communication diagram
    hellorpi =0 #flag for handshaking
    vmstarting = 0
    perfect = 0
    vmstarting = 0
    sp = 0
    coorrec = 0
    motioncompleted = 0
    
    #Initialise communication
    ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    print("[INFO]Serial port opened")
    
    while hellorpi==0:
        line = ser.readline().decode(errors='ignore').strip()
        print(line)
        if line =="HelloRpi" :
            hellorpi=1
            #msg = "HelloSTM\r\n"
            #ser.write(msg.encode())
            while perfect == 0:
                msg = "HelloSTM\r\n"
                ser.write(msg.encode())
                time.sleep(0.2)
                line = ser.readline().decode(errors='ignore').strip()
                print(line)
                if line == 'PERFECT':
                    perfect=1
                    #break
            
    robot = PickAndPlaceRobot(ser)
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        print(line)
        if line == "STARTPROCESS":
            sp=1
            robot.run()
                

if __name__ == "__main__":
    main()
