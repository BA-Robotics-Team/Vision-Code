#this code exists to verify the bounding box coordinates
 
from ultralytics import YOLO
img_path="" #Image path 

Y=YOLO("Models\\Card_Box\\2\\runs\\train\\yolov8_obb_biscuit\\weights\\best.pt")
result=Y.predict(img_path , save=True,show=True,verbose=False)
print(result[0].obb.xyxyxyxy)
print("Normalised: ",result[0].obb.xyxyxyxyn)