import os
try:
    import kagglehub
except ImportError:
    print("Kagglehub not found. Installing...")
    os.system('pip install kagglehub')
    import kagglehub

def download_dataset():
    """
    Downloads the PCB defect dataset and returns the path.
    """
    print("Initializing dataset download from Kaggle...")
    path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
    
    print(f"Dataset downloaded to: {path}")
    print("\nNext Steps:")
    print("1. Ensure 'data.yaml' points to this directory.")
    print("2. Run 'python train.py' to start training (or use pre-trained weights).")
    return path

if __name__ == "__main__":
    download_dataset()
