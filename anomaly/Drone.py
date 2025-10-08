import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import cv2
from ultralytics import YOLO

# Load YOLOv11x model (fine-tuned) once, cached for performance
@st.cache_resource
def load_model():
    model = YOLO("C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/best.pt")

    return model

model = load_model()

st.title("Drone Detection using YOLOv11x")
st.markdown("Upload an image or enter an image URL to detect drones")

def load_image_from_upload(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    return None

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except (requests.exceptions.RequestException, UnidentifiedImageError):
        return None

def detect_drones(img_np):
    results = model(img_np, conf=0.25)
    image_copy = img_np.copy()
    drones = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]
                drones.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })
                # Draw bounding box and label
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image_copy,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
    return image_copy, drones

input_type = st.radio("Select Input Type:", ["Upload Image", "Enter Image URL"])

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_np = load_image_from_upload(uploaded_file)
    else:
        image_np = None
else:
    url = st.text_input("Enter Image URL:")
    if url:
        image_np = load_image_from_url(url)
        if image_np is None:
            st.error("Invalid image URL or unable to load image.")
    else:
        image_np = None

if image_np is not None:
    st.image(image_np, caption="Original Image", use_container_width=True)

if image_np is not None and st.button("Detect Drones"):
    with st.spinner("Detecting drones..."):
        detected_img, detections = detect_drones(image_np)
        st.image(detected_img, caption="Detection Results", use_container_width=True)
        if detections:
            st.markdown("### Detections:")
            for det in detections:
                st.write(f"Label: {det['label']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")
        else:
            st.info("No drones detected.")

if st.button("Clear"):
    st.experimental_rerun()
