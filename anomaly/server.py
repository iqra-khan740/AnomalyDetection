import os
os.environ['ULTRALYTICS_GIT_DISABLE'] = '1'

import cv2
import requests
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
from typing import Optional

app = FastAPI(title="YOLO Person Detection API", description="API for person detection using pretrained and fine-tuned YOLO models")

# Load models at startup
pretrained_model = YOLO("yolov5mu.pt")
fine_tuned_model = YOLO("fine_tuned_model.pt")
PERSON_CLASS_INDEX = 0

def load_image_from_file(file: UploadFile) -> Optional[np.ndarray]:
    """Load image from uploaded file."""
    try:
        contents = file.file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        return np.array(image)
    except Exception:
        return None

def load_image_from_url(url: str) -> Optional[np.ndarray]:
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

def detect_persons(model, image: np.ndarray) -> tuple:
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

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    if image is None:
        return None
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/detect/pretrained")
async def detect_pretrained(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """Detect persons using pretrained model. Provide either file or url."""
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Provide either file or url")
    
    image = load_image_from_file(file) if file else load_image_from_url(url)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    detected_image, count = detect_persons(pretrained_model, image)
    base64_image = image_to_base64(detected_image)
    
    return JSONResponse(content={
        "model": "pretrained",
        "person_count": count,
        "detected_image_b64": base64_image
    })

@app.post("/detect/finetuned")
async def detect_finetuned(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """Detect persons using fine-tuned model. Provide either file or url."""
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Provide either file or url")
    
    image = load_image_from_file(file) if file else load_image_from_url(url)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    detected_image, count = detect_persons(fine_tuned_model, image)
    base64_image = image_to_base64(detected_image)
    
    return JSONResponse(content={
        "model": "finetuned",
        "person_count": count,
        "detected_image_b64": base64_image
    })

@app.post("/detect/compare")
async def detect_compare(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """Compare detection using both models. Provide either file or url."""
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Provide either file or url")
    
    image = load_image_from_file(file) if file else load_image_from_url(url)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Pretrained
    pretrained_image, pretrained_count = detect_persons(pretrained_model, image)
    pretrained_b64 = image_to_base64(pretrained_image)
    
    # Fine-tuned
    finetuned_image, finetuned_count = detect_persons(fine_tuned_model, image)
    finetuned_b64 = image_to_base64(finetuned_image)
    
    return JSONResponse(content={
        "pretrained": {
            "person_count": pretrained_count,
            "detected_image_b64": pretrained_b64
        },
        "finetuned": {
            "person_count": finetuned_count,
            "detected_image_b64": finetuned_b64
        }
    })

@app.get("/")
async def root():
    return {"message": "YOLO Person Detection API is running. Use /docs for interactive API."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)