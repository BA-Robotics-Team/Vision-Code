from ultralytics import YOLO
import torch
# devce=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(devce)
# Load a pre-trained OBB model
model = YOLO('Models\\YOLO_PT\\yolo11n-obb.pt')  # Or yolov8s-obb.pt / yolov8m-obb.pt

# Train the model
model.train(
    data='Models\\Card_Box\\1\\Dataset\\data.yaml',
    epochs=10,
    patience=10,
    imgsz=640,
    device='cpu',   # 0 = first GPU
    workers=2,
    batch=16,
    name='BEx',
    save=True,
    project='Models\\Card_Box\\1\\runs',
    exist_ok=True,)
