from ultralytics import YOLO

Y=YOLO("/content/drive/MyDrive/LOLO/Yolo_Model/runs/LOLOtrain/weights/best.pt")
game=Y.predict("/content/daddi.mp4",save=True)

