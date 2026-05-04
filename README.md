PCB Defect Detection 
Introduction
This project discusses the development of an AI-enabled system that can detect any defect in the printed circuit board automatically. The system takes PCB image input and detects the defects that include mouse bites, hole absence, and open circuits. It is proposed to use artificial intelligence to achieve quality results through efficient inspection.

Objective
The main objective of this project is to automate the PCB inspection process and make it faster and more reliable. It focuses on reducing human errors that occur during manual inspection and improving the accuracy of detecting very small and complex defects. Another goal is to make the system capable of working in real-time so that it can be used in industrial environments.

Problem to Solve
In traditional PCB manufacturing, inspection is done manually, which is a slow and inconsistent process. Human inspectors may miss small defects due to fatigue or limited visibility, especially when working with large volumes of boards. This leads to poor quality products and increased production costs. Therefore, there is a need for an automated system that can detect defects quickly and accurately.

Solution
To solve this problem, the project uses YOLOv8, a modern object detection model that can process images in a single pass and provide fast results. The system is trained using labeled PCB images so that it can recognize different types of defects. When a new image is given as input, the model detects the defect, identifies its type, and marks its location using bounding boxes along with a confidence score.

Technology Used
The project is implemented using Python as the main programming language. YOLOv8 is used as the core detection model. OpenCV is used for image processing tasks, and convolution-based models are used to analyze visual patterns and detect defects from PCB images.

Final Output
The system produces an output image where detected defects are highlighted with bounding boxes and labels. Each detection includes a confidence score indicating how certain the model is about the defect. The system works efficiently and is suitable for real-time inspection. It can be integrated into industrial systems to improve quality control and reduce manual effort.


GUI for the Imput image:

INPUT:
OUTPUT: 
