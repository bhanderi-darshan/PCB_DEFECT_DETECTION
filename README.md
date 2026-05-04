# Industrial PCB Defect Detection System

An AI-powered automated inspection system designed to identify and classify manufacturing defects in Printed Circuit Boards (PCBs) using the **YOLOv8** deep learning architecture.

## 🚀 Features
*   **Real-time Detection**: High-speed inference for industrial assembly lines.
*   **Multiple Defect Classes**: Identifies 6 common PCB defects:
    *   Mouse bite
    *   Spur
    *   Missing hole
    *   Short circuit
    *   Open circuit
    *   Spurious copper
*   **Web Dashboard**: User-friendly Flask interface for image upload and automated report generation.
*   **Industrial Analytics**: Maps detected defects to severity levels and required manufacturing processes.

## 🧠 Project Fundamentals

### How the Model Works (YOLOv8)
The project utilizes **YOLOv8 (You Only Look Once version 8)**, a state-of-the-art Real-Time Object Detection model.
*   **Single-Shot Detection**: YOLO looks at the entire image in one forward pass, making it significantly faster than traditional models.
*   **Architecture**: Uses a Darknet backbone for feature extraction and a specialized head for predicting bounding boxes.
*   **Anchor-Free**: Unlike older versions, YOLOv8 uses anchor-free detection, which allows it to better identify the irregular shapes of PCB defects.

### Why YOLOv8?
*   **Speed vs. Accuracy**: Faster R-CNN is accurate but slow (~5-10 FPS). YOLOv8 provides similar accuracy at much higher speeds (>50 FPS), which is critical for industrial lines.
*   **Small Object Detection**: PCB defects are often tiny. YOLOv8's improved loss functions and feature fusion make it superior at detecting small-scale anomalies compared to SSD or earlier YOLO versions.

## 📊 Evaluation & Metrics
The model is evaluated using standard Computer Vision metrics:
*   **Precision**: Accuracy of the positive predictions.
*   **Recall**: Ability of the model to find all defects in the image.
*   **mAP50**: Mean Average Precision, indicating the overall reliability of the detection engine.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhanderi-darshan/PCB_DEFECT_DETECTION.git
   cd PCB_DEFECT_DETECTION
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset (Optional):**
   ```bash
   python download_data.py
   ```

## 🚀 Running the Application

1. **Start the Flask Web App:**
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`.
3. Upload a PCB image to see the AI analysis in action.

## 📁 Project Structure
*   `app.py`: Main Flask application.
*   `evaluate.py`: Script for model performance validation.
*   `analytics.py`: Industrial logic for severity and process mapping.
*   `templates/`: HTML dashboards for the web interface.
*   `data.yaml`: Dataset configuration file.

## 📄 License
This project is licensed under the MIT License.
