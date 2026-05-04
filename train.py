import os
import torch
from ultralytics import YOLO

# CONFIGURATION
DATA_CONFIG = 'data.yaml'
MODEL_VARIANT = 'yolov8m.pt' # Medium model for High Accuracy
EPOCHS = 50 # Increased epochs for better convergence
BATCH_SIZE = 8
IMG_SIZE = 640
PROJECT_NAME = 'pcb_inspections'

def train_model():
    """
    Train YOLOv8 on the PCB dataset.
    This logic will run on CPU unless CUDA is detected.
    """
    print("Initializing YOLOv8 Training Script...")
    
    # Load base model
    model = YOLO(MODEL_VARIANT)
    
    # Train
    print(f"Starting training on {DATA_CONFIG} for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name='pcb_defect_model',
        device='cpu' # Explicitly use CPU for this request
    )
    
    print("Training Complete!")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    
    # Helper: Copy best weights to models/ folder for app.py
    os.makedirs('models', exist_ok=True)
    best_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, 'models/best.pt')
        print("Copied best weights to /models/best.pt")

if __name__ == "__main__":
    train_model()
