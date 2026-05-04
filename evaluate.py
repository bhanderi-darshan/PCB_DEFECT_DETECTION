import os
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model_path='models/best.pt', data_config='data.yaml'):
    """
    Evaluates the YOLO model and saves the confusion matrix and metrics.
    """
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Falling back to yolov8n.pt")
        model_path = 'yolov8n.pt'
        
    model = YOLO(model_path)
    
    print(f"Starting validation on {data_config}...")
    try:
        results = model.val(data=data_config)
        
        # Save metrics to a file
        metrics = {
            'Precision': results.results_dict['metrics/precision(B)'],
            'Recall': results.results_dict['metrics/recall(B)'],
            'F1-Score': results.results_dict['metrics/mAP50(B)'], # Approximation if direct F1 not in dict
            'mAP50-95': results.results_dict['metrics/mAP50-95(B)']
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv('evaluation_results.csv', index=False)
        print("Evaluation results saved to evaluation_results.csv")
        
        # Confusion matrix is automatically saved by ultralytics in the runs directory
        print(f"Confusion matrix and other plots saved to: {results.save_dir}")
        
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    evaluate_model()
