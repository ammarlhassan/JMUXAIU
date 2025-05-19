import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

# Load trained yolo model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small pretrained model

# streamlit setup
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")
st.title("Real Time Object Detection for Autonomous Driving")

# Display logos of universities
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("/Users/jacknelson/Desktop/obj/jmu_logo.png", width=100)
with col3:
    st.image("/Users/jacknelson/Desktop/obj/alamein_logo.png", width=100)

st.markdown("---")
st.subheader("Live Detection Feed")

# webcam video input
run = st.checkbox('Start Detection')

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(1)

# function to detect and annotate
def detect_and_display(frame):
    results = model(frame)
    annotated = np.squeeze(results.render())
    return annotated

# main loop
while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to access camera.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = detect_and_display(frame)
    FRAME_WINDOW.image(output, channels="RGB")

camera.release()
st.stop()
