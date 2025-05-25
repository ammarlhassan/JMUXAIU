import cv2
import torch
import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Team B Object Detection", layout="wide")
st.title("Real-Time Object Detection for Autonomous Driving")
st.subheader("Team B: Pedestrians, Traffic Cones, Lane Detection")

@st.cache_resource
def load_models():
    pedestrian_model = YOLO('yolov8m.pt')
    lane_model = YOLO('my_model.pt')
    cone_model = YOLO('my_model2.pt')  # Local cone model
    return pedestrian_model, lane_model, cone_model

pedestrian_model, lane_model, cone_model = load_models()

st.sidebar.title("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Upload Video"])

def infer_cones_local(image_np, conf_threshold):
    results = cone_model(image_np, conf=conf_threshold)[0]
    return results

def draw_cone_detections(image_np, results):
    for box, cls in zip(results.boxes, results.boxes.cls):
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Traffic Cone: {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (80, 127, 255), 3)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return image_np

def draw_lane_detections(image_np):
    results = lane_model(image_np, conf=0.25)[0]
    for box, cls in zip(results.boxes, results.boxes.cls):
        if int(cls.item()) != 5:  # Show only class 5, which corresponds to "Lane"
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        label = f"Lane: {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image_np

def process_frame(frame, conf_threshold):
    results = pedestrian_model(frame, conf=conf_threshold)[0]
    person_mask = results.boxes.cls == 0
    results.boxes = results.boxes[person_mask]

    annotated_frame = results.orig_img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        label = f"Pedestrian: {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cone_results = infer_cones_local(frame, conf_threshold)
    annotated_frame = draw_cone_detections(annotated_frame, cone_results)

    annotated_frame = draw_lane_detections(annotated_frame)

    return annotated_frame

if input_source == "Webcam":
    st.subheader("Live Detection Feed")
    run = st.checkbox('Start Detection')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(1)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access camera. Try changing the camera index.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = process_frame(frame, confidence_threshold)
        FRAME_WINDOW.image(output, channels="RGB")

    camera.release()

elif input_source == "Upload Image":
    st.subheader("Image Detection")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_np, channels="RGB")

        with col2:
            st.subheader("Detected Objects")
            result = process_frame(image_np, confidence_threshold)
            st.image(result, channels="RGB")

elif input_source == "Upload Video":
    st.subheader("Video Detection")
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        video_cap = cv2.VideoCapture(temp_file)
        stframe = st.empty()

        play_button = st.button("Play Video Detection")
        stop_button = st.button("Stop")

        if play_button and not stop_button:
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = process_frame(frame_rgb, confidence_threshold)
                stframe.image(result, channels="RGB")
                cv2.waitKey(1)

        video_cap.release()

st.markdown("---")
st.markdown("""
### Team B Detection Classes
- **Pedestrians**: YOLOv8 (`yolov8m.pt`)
- **Traffic Cones**: Local YOLO model (`my_model2.pt`)
- **Traffic Lanes**: YOLOv8 (`my_model.pt`)
""")
