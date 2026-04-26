import os
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import streamlit as st
from ultralytics import YOLO
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from PIL import Image
import tempfile

import cv2
cv2.ocl.setUseOpenCL(False)  # Disable GPU acceleration

# Load model efficiently and cache it so it doesn't reload on every UI interaction
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar for settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Page title
st.title("Flood Rescue Detection System")

# Description
st.write("Upload an image or video to detect: `car`, `person`, `in-car`, and `drowning`.")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

def process_results(results):
    """Counts classes and checks for drowning"""
    counts = {"car": 0, "person": 0, "in-car": 0, "drowning": 0}
    drowning_alert = False
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Check for our valid class names
                if class_name in counts:
                    counts[class_name] += 1
                if class_name == "drowning":
                    drowning_alert = True
                
    return counts, drowning_alert

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type in ["jpg", "jpeg", "png"]:
        # Process Image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Run YOLO detection
        results = model(image_np, conf=conf_threshold)
        
        # Get counts and alerts
        counts, is_drowning = process_results(results)
        
        if is_drowning:
            st.error("🚨 URGENT: Drowning person detected! Immediate assistance required. 🚨", icon="🆘")
        elif counts.get("in-car", 0) > 0:
            st.warning("⚠️ Warning: People detected inside a car.", icon="⚠️")
        
        # Display analytics
        st.subheader("Detection Summary")
        cols = st.columns(4)
        cols[0].metric("People", counts.get("person", 0))
        cols[1].metric("Cars", counts.get("car", 0))
        cols[2].metric("People in Car", counts.get("in-car", 0))
        cols[3].metric("Drowning", counts.get("drowning", 0))
        
        # Draw bounding boxes
        annotated_image = results[0].plot()
        
        st.subheader("Visual Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
            
    elif file_type in ["mp4", "avi", "mov"]:
        # Process Video
        # Save video to temporary file to process with OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") 
        tfile.write(uploaded_file.read())
        
        vid_cap = cv2.VideoCapture(tfile.name)
        
        # Create placeholders in UI for updating frames and metrics
        st_frame = st.empty()
        
        # Metrics placeholders
        st.subheader("Live Detection Summary")
        col1, col2, col3, col4 = st.columns(4)
        metric_person = col1.empty()
        metric_car = col2.empty()
        metric_incar = col3.empty()
        metric_drowning = col4.empty()
        
        # Alert placeholder
        alert_placeholder = st.empty()
        
        stop_button = st.button("Stop Video Processing")
        
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret or stop_button:
                break
                
            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO
            results = model(frame_rgb, conf=conf_threshold)
            counts, is_drowning = process_results(results)
            
            # Update metrics
            metric_person.metric("People", counts.get("person", 0))
            metric_car.metric("Cars", counts.get("car", 0))
            metric_incar.metric("People in Car", counts.get("in-car", 0))
            metric_drowning.metric("Drowning", counts.get("drowning", 0))
            
            # Update alerts
            if is_drowning:
                alert_placeholder.error("🚨 URGENT: Drowning person detected! Immediate assistance required. 🚨", icon="🆘")
            elif counts.get("in-car", 0) > 0:
                alert_placeholder.warning("⚠️ Warning: People detected inside a car.", icon="⚠️")
            else:
                alert_placeholder.empty()
                
            # Plot results on frame
            annotated_frame = results[0].plot()
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
            
        vid_cap.release()