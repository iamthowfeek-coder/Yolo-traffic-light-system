import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# Initialize YOLO models
model_traffic = YOLO('best.pt')  # YOLO model for traffic detection
model_emergency1 = YOLO('amb.pt')  # YOLO model for emergency vehicles
model_emergency2 = YOLO('amb1.pt')  # Second model for emergency vehicles

# Class names for traffic and emergency vehicles
classnames = ['bus', 'car', 'motorbike', 'truck', 'van']
emergency_classnames = ['ambulance']
done_classnames = ['Ambulance']

# Load traffic light images
green_light_img = cv2.imread('green.jpg')
red_light_img = cv2.imread('red.jpg')
green_light_img = cv2.resize(green_light_img, (100, 100))
red_light_img = cv2.resize(red_light_img, (100, 100))

# Function to process frames
def process_frame(frame):
    vehicles_count = {'bus': 0, 'car': 0, 'motorbike': 0, 'truck': 0, 'van': 0}
    emergency_count = {'ambulance': 0}

    frame = cv2.resize(frame, (640, 480))

    # Process with traffic model
    traffic_results = model_traffic(frame)
    for info in traffic_results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if confidence > 0.5 and class_id < len(classnames):
                vehicles_count[classnames[class_id]] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{classnames[class_id]}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Process with emergency vehicle models
    for model in [model_emergency1, model_emergency2]:
        emergency_results = model(frame)
        for info in emergency_results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                if confidence > 0.5:
                    emergency_count['ambulance'] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'Ambulance', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, vehicles_count, emergency_count

# Function to manage traffic lights
def manage_traffic_lights(vehicles_count, emergency_count, frame):
    total_vehicles = sum(vehicles_count.values())
    total_emergency = sum(emergency_count.values())

    if total_emergency > 0 and total_vehicles <= 25:
        frame[10:110, 10:110] = red_light_img
        frame[10:110, 120:220] = red_light_img
        frame[10:110, 230:330] = green_light_img
    elif total_emergency > 0:
        frame[10:110, 10:110] = red_light_img
        frame[10:110, 120:220] = red_light_img
        frame[10:110, 230:330] = red_light_img
    elif total_vehicles <= 25:
        frame[10:110, 10:110] = green_light_img
        frame[10:110, 120:220] = red_light_img
        frame[10:110, 230:330] = red_light_img
    else:
        frame[10:110, 10:110] = red_light_img
        frame[10:110, 120:220] = green_light_img
        frame[10:110, 230:330] = red_light_img

    return frame

# Sign-up page
def sign_up():
    st.title("🚀 Sign-Up Page")
    st.write("Join our community and manage traffic efficiently! 🌟")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type='password')
    email = st.text_input("📧 Email")
    if st.button("Sign Up"):
        st.success(f"Welcome {username}! Your account has been created successfully. 🎉")

# Login page
def login():
    st.title("🔑 Login Page")
    st.write("Access the traffic management system with your credentials.")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type='password')
    if st.button("Login"):
        st.success(f"Welcome back, {username}! 🚗")

# Traffic management app
def traffic_management_app():
    st.title('🚦 Traffic Flow Management with Emergency Detection')
    uploaded_file = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, count, emergency_count = process_frame(frame)
            processed_frame = manage_traffic_lights(count, emergency_count, processed_frame)

            total_vehicles = sum(count.values())
            total_emergency = sum(emergency_count.values())

            text = f'Total Vehicles: {total_vehicles} | Ambulances: {total_emergency}'
            cv2.putText(processed_frame, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

        cap.release()

# Home page
def home_page():
    st.title("🏠 Home Page")
    st.write("Welcome to the **Traffic Flow Management System**! 🎉 This app detects traffic and emergency vehicles in real-time.")
    st.markdown("""
    ### 🚗 Features:
    - **Real-time traffic detection** 🕒
    - **Emergency vehicle prioritization** 🚑
    - **Dynamic traffic light management** 🚦
    
    ### 🌟 Why Choose Us?
    - **Advanced AI Models** 🤖
    - **Accurate Traffic Flow Predictions** 📊
    - **Seamless User Experience** 🎯
    """)
    st.write("🚀 Start managing traffic with precision and efficiency!")

# Main application
def main():
    menu = ["Home", "Sign Up", "Login", "Traffic Prediction"]
    choice = st.sidebar.selectbox("🌟 Navigation", menu)

    if choice == "Home":
        home_page()
    elif choice == "Sign Up":
        sign_up()
    elif choice == "Login":
        login()
    elif choice == "Traffic Prediction":
        traffic_management_app()

if __name__ == "__main__":
    main()
