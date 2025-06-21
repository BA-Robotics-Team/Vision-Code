from ultralytics import YOLO
Y=YOLO("runs\\train\\yolov8_obb_biscuit\\weights\\best.pt")
result=Y.predict("50NEW\\Valid\\Images\\frame_00476.jpg" , save=True,show=True,verbose=False)
print(result[0].obb.xyxyxyxy)
print("Normalised: ",result[0].obb.xyxyxyxyn)