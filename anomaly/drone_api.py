from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import io
import requests

# ------------------------------
# Initialize FastAPI App
# ------------------------------
app = FastAPI(
    title="Drone Detection API",
    description="Upload or provide an image URL to detect drones using YOLOv11x",
    version="1.0"
)

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (customize for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load YOLOv11x Model (Once)
# ------------------------------
model_path = "C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/best.pt"
model = YOLO(model_path)


# ------------------------------
# Helper Functions
# ------------------------------
def load_image_from_upload(uploaded_file: UploadFile):
    try:
        image = Image.open(io.BytesIO(uploaded_file.file.read())).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")


def load_image_from_url(url: str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            raise HTTPException(status_code=400, detail="URL does not contain an image.")
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Invalid or unreachable image URL.")


def detect_drones(img_np: np.ndarray):
    results = model(img_np, conf=0.25)
    image_copy = img_np.copy()
    drones = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                drones.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })
                # Draw bounding box
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


# ------------------------------
# API Endpoint: Detect Drones
# ------------------------------
@app.post("/detect")
async def detect_drone(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    return_image: bool = Form(False)
):
    """
    Upload an image file OR provide an image URL for drone detection.
    Optionally, set return_image=True to get annotated image.
    """

    if file is None and image_url is None:
        raise HTTPException(status_code=400, detail="Please upload an image or provide an image URL.")

    # Load image
    if file:
        image_np = load_image_from_upload(file)
    else:
        image_np = load_image_from_url(image_url)

    # Detect drones
    detected_img, detections = detect_drones(image_np)

    # Return image if requested
    if return_image:
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR))
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    # Return JSON by default
    return JSONResponse({
        "status": "success",
        "detections": detections,
        "total_drones_detected": len(detections)
    })


# ------------------------------
# Root Endpoint
# ------------------------------
@app.get("/")
def root():
    return {"message": "🚁 Drone Detection API is running. Use /docs to test."}


# ------------------------------
# Run Command
# ------------------------------
# Run in terminal:
# uvicorn drone_api:app --reload