import cv2
import numpy as np
from ultralytics import YOLO
from workspace_kinematics import IKinematics
from serial import Serial
from threading import Thread
import math as m

# Constants
WORKSPACE_WIDTH = 17.1  # cm 17.25
WORKSPACE_HEIGHT = 12      # cm 11.4

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
        if len(squares) == 4:break
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
        print(bottom)
        return [bottom[0], bottom[1], top[1], top[0]]  # bl, br, tr, tl
    return []

def objectDetect(frame,model):
    result=model.predict(frame)
    resultIMG=result[0].plot()
    resultBox=result[0].boxes
    if not resultBox:
        return [0,0], resultIMG
    for box in resultBox:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Top-left (x1, y1) and bottom-right (x2, y2) coordinates
        
        # Calculate the center of the box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return [center_x,center_y],resultIMG

def overlay_border(img, pts,Ocoords):
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
    
    #this marks the obj coordinates
    coord=((Ocoords[0] - origin[0]) / px_per_cm_x, -1*(Ocoords[1] - origin[1]) / px_per_cm_y)
    cv2.putText(img, f"({coord[0]:.1f},{coord[1]:.1f})", (Ocoords[0]+5, Ocoords[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    print("BottomLeft:",bl)
    print("BottomRight:",br)
    print("TopLeft:",tl)
    print("TopRight:",tr)
    
    return img

def IKinematics(ObjCoords):
 link1,link2,x,y=12,12,ObjCoords[0],ObjCoords[1] #constants, final position
 base=m.atan2(y,x)*(180/m.pi) #base motor/M0
 w=(x**2 +y**2)**0.5
 alpha=m.acos((link1**2 + link2**2 - w**2)/(2*link1*link2))*(180/m.pi)
 motor2=180-alpha #M2
 motor1=m.acos((w**2 + link1**2 - link2**2)/(2*w*link1))*(180/m.pi)#M1
 beta=m.acos((w**2 + link2**2 - link1**2)/(2*w*link2))*(180/m.pi)
 return [base,motor1,motor2]

def  Serial_comm(port,baudrate,motor_parameters):
    with Serial(f"COM{port}", baudrate, timeout=0.2) as ser:
     ser.write(f"{motor_parameters[0]},{motor_parameters[1]},{motor_parameters[2]}".encode())

def run():
    cap = cv2.VideoCapture(1)
    model=YOLO("Dbest.pt")
    
    for x in range(5):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, -1)
        mask = get_black_mask(frame)
        squares = detect_squares(mask)
        corners = get_inner_corners(squares, np.array(frame.shape[1::-1]) // 2)
       
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, -1)        
        objCoords,resultIMG=objectDetect(frame,model) 
        if objCoords:
          out = overlay_border(resultIMG, corners,objCoords)
          actuations=IKinematics(objCoords)
          Serial_comm(3,9600,actuations)
        cv2.imshow("Workspace", out)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

run()

