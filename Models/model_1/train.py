from ultralytics import YOLO 
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset path
DATA_YAML_PATH = "Dataset\\data.yaml"  # Update if needed

# Initialize YOLOv11-m model                                                    
model = YOLO("diff_files\\yolo11n.pt")  # Load pre-trained YOLOv11-m weights

#Train the model
model.train(
    data=DATA_YAML_PATH,  # Path to dataset config
    epochs=1,           # More epochs for better convergence
    imgsz=640,            # Standard YOLO image size
    batch=16,             # Balanced batch size for 3,300 images
    device=device, # Use GPU if available
    workers=2,            # Optimize for Kaggle's CPU cores
    save=True,            # Save model checkpoints
    project="results",  # Save results in Kaggle workspace
    name="result",
    patience=5,          # Early stopping if no improvement in 10 epochs
    optimizer="AdamW",    # Adaptive optimizer for better training stability
    lr0=0.001,            # Initial learning rate
    weight_decay=0.0005,  # Regularization to prevent overfitting
)

print("Training completed")