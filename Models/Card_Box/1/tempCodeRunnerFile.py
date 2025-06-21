from ultralytics import YOLO
Y=YOLO("runs\\BEx\\weights\\best.pt")
result=Y.predict("Models\\Card_Box\\1\\Dataset\\Valid\\images\\frame_00598.jpg" , save=True,show=True,verbose=False)
print(result[0].obb.xyxyxyxy)
print("Normalised: ",result[0].obb.xyxyxyxyn)