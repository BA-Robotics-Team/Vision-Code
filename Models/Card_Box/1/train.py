from ultralytics import YOLO

# Load a pre-trained OBB model
model = YOLO('yolo11n-obb.pt')  # Or yolov8s-obb.pt / yolov8m-obb.pt

# Train the model
model.train(
    data='Biscuit\\data.yaml',
    epochs=10,
    patience=10,
    imgsz=640,
    device='cpu',   # 0 = first GPU
    workers=4,
    batch=16,
    name='yolov8_obb_biscuit',
    save=True,
    project='runs/train',
    exist_ok=True,

)
