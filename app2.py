import cv2
import torch
import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

st.set_page_config(page_title="Team B Object Detection", layout="wide")
st.title("Real-Time Object Detection for Autonomous Driving")
st.subheader("Team B: Pedestrians, Traffic Cones, Lane Detection")

@st.cache_resource
def load_models():
    pedestrian_model = YOLO('yolov8m.pt')
    lane_model = YOLO('my_model.pt')
    return pedestrian_model, lane_model

pedestrian_model, lane_model = load_models()

CLIENT = InferenceHTTPClient(
    api_url = "https://detect.roboflow.com",
    api_key = api_key
)

st.sidebar.title("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Upload Video"])

def infer_cones_with_roboflow(image_np):
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    result = CLIENT.infer(temp_path, model_id="cone-6un5f/2")
    return result

def draw_roboflow_results(image_np, results):
    for pred in results["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        confidence = pred["confidence"]
        class_name = pred["class"]

        if class_name.lower() == "cone":
            class_name = "Traffic Cone"

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (80, 127, 255), 3)
        cv2.putText(image_np, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return image_np

def draw_lane_detections(image_np):
    results = lane_model(image_np, conf=0.25)[0]
    for box in results.boxes:
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

    roboflow_result = infer_cones_with_roboflow(frame)
    annotated_frame = draw_roboflow_results(annotated_frame, roboflow_result)

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

        video_cap.release()

st.markdown("---")
st.markdown("""
### Team B Detection Classes
- **Pedestrians**: YOLOv8
- **Traffic Cones**: Roboflow API (cone-6un5f/2)
- **Traffic Lanes**: YOLOv8 (my_model.pt)
""")
