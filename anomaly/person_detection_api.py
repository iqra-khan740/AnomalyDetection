from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import io
import requests

# ------------------------------------------
# Initialize FastAPI App
# ------------------------------------------
app = FastAPI(
    title="YOLO Person Detection API",
    description="Compare Pretrained and Fine-tuned YOLO models for person detection",
    version="1.0",
)

# Enable CORS (optional, useful for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# Load Models (Once)
# ------------------------------------------
PRETRAINED_PATH = "C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/yolov5mu.pt"
FINETUNED_PATH = "C:/Users/DELL/OneDrive/Desktop/AnomalyDetectio/AnomalyModelfiles/fine_tuned_model.pt"

pretrained_model = YOLO(PRETRAINED_PATH)
fine_tuned_model = YOLO(FINETUNED_PATH)
PERSON_CLASS_INDEX = 0  # 'person' class


# ------------------------------------------
# Helper Functions
# ------------------------------------------
def load_image_from_upload(uploaded_file: UploadFile):
    """Load image from uploaded file."""
    try:
        image = Image.open(io.BytesIO(uploaded_file.file.read())).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")


def load_image_from_url(url: str):
    """Load image from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            raise HTTPException(status_code=400, detail="URL does not contain an image.")
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Invalid or unreachable image URL.")


def detect_persons(model, image_np):
    """Run YOLO model to detect persons."""
    image_resized = cv2.resize(image_np, (640, 640))
    results = model(image_resized, conf=0.25)
    head_count = 0

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls == PERSON_CLASS_INDEX:
                    head_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_resized, head_count


# ------------------------------------------
# API Endpoint
# ------------------------------------------
@app.post("/detect")
async def detect_person(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    return_images: bool = Form(False)
):
    """
    Upload an image or provide an image URL to detect persons using two YOLO models.
    Set return_images=True to get both annotated images as output.
    """

    # Check inputs
    if file is None and image_url is None:
        raise HTTPException(status_code=400, detail="Please upload an image or provide an image URL.")

    # Load image
    if file:
        image_np = load_image_from_upload(file)
    else:
        image_np = load_image_from_url(image_url)

    # Run both models
    pretrained_img, pretrained_count = detect_persons(pretrained_model, image_np)
    finetuned_img, finetuned_count = detect_persons(fine_tuned_model, image_np)

    # If returning images
    if return_images:
        # Combine both images side-by-side
        combined = np.hstack((pretrained_img, finetuned_img))
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    # Return JSON
    return JSONResponse({
        "status": "success",
        "results": {
            "pretrained": {
                "person_count": pretrained_count,
            },
            "fine_tuned": {
                "person_count": finetuned_count,
            },
            "comparison": {
                "difference": finetuned_count - pretrained_count
            }
        }
    })


# ------------------------------------------
# Root Endpoint
# ------------------------------------------
@app.get("/")
def root():
    return {"message": "👤 YOLO Person Detection API is running. Use /docs to test."}


# ------------------------------------------
# Run Command
# ------------------------------------------
# Run in terminal (inside your project folder):
# uvicorn person_detection_api:app --reload
