import os
import time
import uuid
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from ultralytics import YOLO
from analytics import analyze_defects
from comparison import align_images, detect_differences

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "pcb_secret_key"

# Configure Paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'models/best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global Model Variable
model = None
model_status = "Initializing..."

def load_ai_model():
    """Load the YOLOv8 model safely."""
    global model, model_status
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            model_status = "Expert PCB AI (best.pt) Loaded"
            print(f"Success: {model_status}")
        except Exception as e:
            print(f"Error loading {MODEL_PATH}: {e}")
            model = YOLO('yolov8n.pt')
            model_status = "Fallback: Generic Model"
    else:
        print("Warning: Trained 'best.pt' not found. Falling back to base YOLOv8n.")
        model = YOLO('yolov8n.pt')
        model_status = "Fallback: Generic Model"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle Upload and Detection logic."""
    if 'sample' not in request.files:
        flash("Sample image is required.")
        return redirect(url_for('index'))
    
    sample_file = request.files['sample']
    
    if sample_file.filename == '':
        flash("No sample file selected.")
        return redirect(url_for('index'))

    # Save Sample
    sample_filename = f"sample_{uuid.uuid4().hex}_{sample_file.filename}"
    sample_path = os.path.join(UPLOAD_FOLDER, sample_filename)
    sample_file.save(sample_path)
    
    # 3. RUN AI INFERENCE (YOLO)
    if model is None:
        load_ai_model()
        
    start_time = time.time()
    results = model.predict(source=sample_path, conf=0.25, save=False, imgsz=640)
    inference_time = round(time.time() - start_time, 2)
    
    # 4. ANALYZE RESULTS (YOLO only)
    analysis = analyze_defects(results)
    analysis['inference_time'] = inference_time
    
    # 5. GENERATE FINAL REPORT IMAGE
    res = results[0]
    img = cv2.imread(sample_path)
    for box in res.boxes:
        b = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = res.names[cls]
        
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (b[0], b[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    result_filename = f"ai_res_{sample_filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, img)
    
    return render_template('report.html', 
                           analysis=analysis,
                           sample_img=sample_filename,
                           result_img=result_filename,
                           model_status=model_status)

if __name__ == "__main__":
    load_ai_model()
    app.run(host='0.0.0.0', port=5000, debug=True)