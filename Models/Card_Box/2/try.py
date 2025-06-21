from ultralytics import YOLO
import math as m
import cv2
import time



Y=YOLO("Models\\Card_Box\\2\\runs\\train\yolov8_obb_biscuit\\weights\\best.pt")

# Y=YOLO("best.pt")
result=Y.predict(1,save=True,show=True)
roll=None
deg=None
for resulta in result:
    for obb in resulta.obb.xywhr:
        roll=obb[4].item()
        deg=m.degrees(roll)


with open("file.txt",'a') as f:
    f.write(str(deg))
    f.write("\n")




    
     