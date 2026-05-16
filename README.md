🚦 YOLO-Based Density-Aware Traffic Light System
An intelligent traffic light control system that uses real-time vehicle density detection to dynamically manage traffic signal timings — reducing congestion and improving traffic flow.
📌 Project Overview
Traditional traffic lights operate on fixed timers, regardless of actual traffic conditions. This project solves that problem by using YOLOv5 object detection to count vehicles in real-time and automatically adjust green light duration based on lane density.
This has direct real-world applications in smart city infrastructure, urban traffic management, and autonomous road systems.
🛠️ Technologies Used
Tools:Python,YOLOV8(You Only Look Once),OpenCV,NumPy
Purpose:Core programming language,Real-time vehicle detection,Video/image processing,Numerical computations
🧠 How It Works
Input — Video feed or image frames of a traffic intersection
Detection — YOLO model detects and counts vehicles in each lane
Density Calculation — System calculates vehicle density per lane
Signal Control — Traffic light timing is dynamically adjusted based on density
Output — Optimized signal timings that prioritize high-density lanes
📂 Dataset
Source: Online dataset (Kaggle)
Contains traffic camera footage and annotated vehicle images
Used for training and testing the YOLO detection model
🚀 How to Run
# Clone the repository
git clone https://github.com/iamthowfeek-coder/Yolo-traffic-light-system.git

# Navigate to the project folder
cd yolo-traffic-light-system

# Install required libraries
pip install -r requirements.txt

# Run the main script
python main.py
📁 Project Structure
yolo-traffic-light-system/
│
├── main.py               # Main script
├── detector.py           # YOLO detection logic
├── traffic_controller.py # Signal timing logic
├── requirements.txt      # Dependencies
├── dataset/              # Sample dataset
└── README.md             # Project documentation
🌍 Real-World Use Case
This system is directly applicable to:
Smart city traffic management (e.g., Dubai RTA initiatives)
Reducing vehicle idle time and carbon emissions
Emergency vehicle prioritization
Autonomous intersection management
👨‍💻 Author
Your Name Kother Thowfeek J
MSc Computer Science
📧 your.email@gmail.com
iamthowfeek@gmail.com
🔗 LinkedIn Profile
https://www.linkedin.com/in/i-am-thowfeek-1739b22b1
🔗 GitHub
git clone https://github.com/iamthowfeek-coder/Yolo-traffic-light-system.git
📜 License
This project is open source and available under the MIT License






