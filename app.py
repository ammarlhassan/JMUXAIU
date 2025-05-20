import cv2
import torch
import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

from dotenv import load_dotenv
load_dotenv() #secure API key load
api_key = os.getenv("ROBOFLOW_API_KEY")



# streamlit page headers
st.set_page_config(page_title="Team B Object Detection", layout="wide")
st.title("Real-Time Object Detection for Autonomous Driving")
st.subheader("Team B: Pedestrians, Traffic Cones")

# Load up YOLOv8 model
@st.cache_resource
def load_models():
    pedestrian_model = YOLO('yolov8m.pt')  # Load COCO-trained pedestrian model
    return pedestrian_model

pedestrian_model = load_models()

# roboflow client setup
CLIENT = InferenceHTTPClient(
    api_url = "https://detect.roboflow.com",
    api_key = api_key # credits will run out new acct needed
)

# settings
st.sidebar.title("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05) #range 0.0-1.1, starts at 0.5

# select input source w/ radio button
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Upload Video"])

# roboflow cone detection
def infer_cones_with_roboflow(image_np):
    temp_path = "temp_frame.jpg" # OpenCV uses BGR, translate image to expected format, and write it to set temp path
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    result = CLIENT.infer(temp_path, model_id="cone-6un5f/2") # call inference on client, giving a file and model ID
    return result

# Result Format:
# {
#   "predictions": [  # List of detected objects in the image
#     {
#       "x": float,          # Center x coordinate of the detected object in pixels (from left)
#       "y": float,          # Center y coordinate of the detected object in pixels (from top)
#       "width": float,      # Width of bounding box in pixels
#       "height": float,     # Height of bounding box in pixels
#       "confidence": float, # Confidence score of the detection (0.0 - 1.0)
#       "class": str         # Detected class label (e.g., "cone", "person", etc.)
#     },
#     # ... more detected objects
#   ],
#   # ... possibly other metadata fields, for example:
#   # "inference_time": float,  # Time taken for inference in seconds
#   # "model_version": str      # Model version or name
# }



def draw_roboflow_results(image_np, results):
    for pred in results["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        confidence = pred["confidence"]
        class_name = pred["class"]

        # rename cone to "Traffic Cone"
        if class_name.lower() == "cone":
            class_name = "Traffic Cone"

        x1, y1 = int(x - w / 2), int(y - h / 2) # top left corner
        x2, y2 = int(x + w / 2), int(y + h / 2) # bottom right corner

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (80, 127, 255), 3) # put rect (color is BGR format)
        cv2.putText(image_np, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return image_np # return new image


# Function to process frames with models
def process_frame(frame, conf_threshold):
    results = pedestrian_model(frame, conf=conf_threshold)[0]

    # Filter only 'person' class (class 0)
    person_mask = results.boxes.cls == 0
    results.boxes = results.boxes[person_mask]

    # Annotate with modified label
    annotated_frame = results.orig_img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        label = f"Pedestrian: {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add Roboflow cone detection
    roboflow_result = infer_cones_with_roboflow(frame)
    annotated_frame = draw_roboflow_results(annotated_frame, roboflow_result)

    return annotated_frame



# Main content based on input source selection
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

# Detection class information
st.markdown("---")
st.markdown("""
### Team B Detection Classes
- **Pedestrians**: YOLOv8
- **Traffic Cones**: Roboflow API (cone-6un5f/2)
""")
