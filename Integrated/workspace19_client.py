import cv2
import numpy as np
from ultralytics import YOLO
from Workspace.kinematicsa import Ikinematics,TCP_Comm
import math as m
import socket
from threading import Thread

# Constants
WORKSPACE_WIDTH = 17.1 # cm
WORKSPACE_HEIGHT= 12   # cm 

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
    if len(pts) != 4: return img,[0,0]
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
    return img,coord

def run():
    s=socket.socket()
    prev=True #variable for tcp control
    cap = cv2.VideoCapture(0)
    model=YOLO("best.pt")
    
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
          out,robjCoords = overlay_border(resultIMG, corners,objCoords)
          try:
           actuations=Ikinematics(robjCoords[0],robjCoords[1],0)
           if prev:
               TCP_Comm(s,actuations)
               prev = False

               ack_sock = socket.socke
               ack_sock.connect(('192.168.31.99', 12345))
               ack = ack_sock.recv(1024).decode()
               if ack:
                   prev = True
               ack_sock.close()
          except: 
             pass
        
        cv2.imshow("Workspace", out)
        if cv2.waitKey(1) &0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

run()