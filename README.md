# Industrial PCB Defect Detection System

📌 **Introduction**
This project discusses the development of an AI-enabled system that can detect any defect in the printed circuit board automatically. The system takes PCB image input and detects the defects that include mouse bites, hole absence, and open circuits. It is proposed to use artificial intelligence to achieve quality results through efficient inspection.

🎯 **Objective**
The main objective of this project is to automate the PCB inspection process and make it faster and more reliable. It focuses on reducing human errors that occur during manual inspection and improving the accuracy of detecting very small and complex defects. Another goal is to make the system capable of working in real-time so that it can be used in industrial environments.

⚠️ **Problem to Solve**
In traditional PCB manufacturing, inspection is done manually, which is a slow and inconsistent process. Human inspectors may miss small defects due to fatigue or limited visibility, especially when working with large volumes of boards. This leads to poor quality products and increased production costs. Therefore, there is a need for an automated system that can detect defects quickly and accurately.

💡 **Solution**
To solve this problem, the project uses **YOLOv8**, a modern object detection model that can process images in a single pass and provide fast results. The system is trained using labeled PCB images so that it can recognize different types of defects. When a new image is given as input, the model detects the defect, identifies its type, and marks its location using bounding boxes along with a confidence score.

---

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

## 🛠️ Technology Used
*   **Python**: Core programming language.
*   **YOLOv8**: Deep learning detection engine.
*   **OpenCV**: Image processing and visualization.
*   **Flask**: Web interface for industrial reports.

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

## 🚀 Running the Application

1. **Start the Flask Web App:**
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`.

---

## 🖼️ Interface & Output

### GUI for Input Image:
<img width="1205" height="878" alt="GUI" src="https://github.com/user-attachments/assets/ab2f070e-c7f6-4f34-aaf1-31aea49b60d5" />

### AI Output Sample:
<img width="566" height="875" alt="Screenshot" src="https://github.com/user-attachments/assets/11e63ce1-57f3-4e74-92b0-8362118a6b5b" />

