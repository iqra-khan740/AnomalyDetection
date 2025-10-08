import streamlit as st
import cv2
import requests
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

# Load models at startup (cached for performance)
@st.cache_resource
def load_models():
    pretrained_model = YOLO("C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/yolov5mu.pt")
    fine_tuned_model = YOLO("C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/fine_tuned_model.pt")

    return pretrained_model, fine_tuned_model

pretrained_model, fine_tuned_model = load_models()
PERSON_CLASS_INDEX = 0

def load_image_from_upload(uploaded_file):
    """Load image from uploaded file."""
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    return None

def load_image_from_url(url):
    """Load image from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except (requests.exceptions.RequestException, UnidentifiedImageError):
        return None

def detect_persons(model, image):
    """Run YOLO model to detect persons."""
    if image is None:
        return None, 0
    image_resized = cv2.resize(image, (640, 640))
    results = model(image_resized, conf=0.25)
    head_count = 0
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id == PERSON_CLASS_INDEX:
                    head_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_resized, head_count

# Streamlit UI
st.title("YOLO Person Detection - Model Comparison")
st.markdown("Upload an image or enter a URL to detect and count persons using pretrained and fine-tuned YOLO models.")

# Input options
input_type = st.radio("Select Input Type:", ["Upload Image", "Enter Image URL"])

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_image_from_upload(uploaded_file)
        original_image = Image.fromarray(image)
        st.image(original_image, caption="Original Image", use_container_width=True)
    else:
        image = None
else:
    url = st.text_input("Enter Image URL:", placeholder="https://example.com/image.jpg")
    if url:
        image = load_image_from_url(url)
        if image is not None:
            original_image = Image.fromarray(image)
            st.image(original_image, caption="Original Image", use_container_width=True)
        else:
            st.error("Invalid URL or image. Please try another.")
            image = None
    else:
        image = None

if image is not None and st.button("Detect Persons"):
    with st.spinner("Running detection..."):
        # Pretrained model
        pretrained_image, pretrained_count = detect_persons(pretrained_model, image)
        st.subheader("Pretrained Model Results")
        if pretrained_image is not None:
            st.image(pretrained_image, caption=f"Detected {pretrained_count} persons", use_container_width=True)
        st.write(f"**Person Count:** {pretrained_count}")

        # Fine-tuned model
        finetuned_image, finetuned_count = detect_persons(fine_tuned_model, image)
        st.subheader("Fine-Tuned Model Results")
        if finetuned_image is not None:
            st.image(finetuned_image, caption=f"Detected {finetuned_count} persons", use_container_width=True)
        st.write(f"**Person Count:** {finetuned_count}")

        # Comparison
        st.subheader("Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pretrained Count", pretrained_count)
        with col2:
            st.metric("Fine-Tuned Count", finetuned_count)

if st.button("Clear"):
    st.rerun()

# Sidebar info
with st.sidebar:
    st.info("This app uses YOLOv5 models for person detection. Green boxes highlight detected persons. The fine-tuned model is optimized for better accuracy on people.")