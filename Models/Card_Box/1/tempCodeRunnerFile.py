from ultralytics import YOLO
Y=YOLO("Workspace\\Weights\\best_gear.pt")
result=Y.predict(1,show=True,verbose=False)

