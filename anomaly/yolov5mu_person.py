import cv2
import requests
import numpy as np
import gradio as gr
from io import BytesIO
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

# Load models
pretrained_model = YOLO("yolov5mu.pt")
fine_tuned_model = YOLO("fine_tuned_model.pt")  # Update with your fine-tuned model path
PERSON_CLASS_INDEX = 0  # Update if needed

def load_image(image, url):
    """Load image from file upload or URL."""
    if image is not None:
        return np.array(image)
    if url:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            if "image" not in response.headers.get("Content-Type", ""):
                return None
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return np.array(image)
        except (requests.exceptions.RequestException, UnidentifiedImageError):
            return None
    return None

def detect_persons(model, image):
    """Run YOLO model to detect persons."""
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

def process_image(image, url, input_type):
    """Process image using both models and compare results."""
    image = load_image(image, None) if input_type == "Upload Image" else load_image(None, url)
    if image is None:
        return None, None, "Invalid image. Please check the URL or upload a valid image.", ""
    image_pretrained, count_pretrained = detect_persons(pretrained_model, image)
    image_finetuned, count_finetuned = detect_persons(fine_tuned_model, image)
    return image_pretrained, image_finetuned, f"Pretrained Model: {count_pretrained}", f"Fine-Tuned Model: {count_finetuned}"

def toggle_input(input_type):
    """Show/hide fields based on input type."""
    return (gr.update(visible=True), gr.update(visible=False)) if input_type == "Upload Image" else (gr.update(visible=False), gr.update(visible=True))

def clear_fields():
    """Clear all inputs and outputs."""
    return None, "", None, None, "", ""
# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("## Person Detection - Model Comparison")
    gr.Markdown("Compare results of a pretrained YOLO model and a fine-tuned YOLO model.")
    input_type = gr.Radio(["Upload Image", "Enter Image URL"], label="Select Input Type", value="Upload Image")
    image_input = gr.Image(type="pil", label="Upload Image", visible=True, height=640, width=640)
    url_input = gr.Textbox(label="Enter Image URL", placeholder="https://example.com/image.jpg", visible=False)
    input_type.change(toggle_input, inputs=[input_type], outputs=[image_input, url_input])
    with gr.Row():
        image_output_pretrained = gr.Image(type="numpy", label="Pretrained Model Output", height=640, width=640)
        image_output_finetuned = gr.Image(type="numpy", label="Fine-Tuned Model Output", height=640, width=640)
    with gr.Row():
        count_pretrained_text = gr.Textbox(label="Pretrained Model Detection Count")
        count_finetuned_text = gr.Textbox(label="Fine-Tuned Model Detection Count")
    with gr.Row():
        clear_button = gr.Button("Clear")
        submit_button = gr.Button("Detect Persons")
    submit_button.click(
        process_image,
        inputs=[image_input, url_input, input_type],
        outputs=[image_output_pretrained, image_output_finetuned, count_pretrained_text, count_finetuned_text]
    )
    clear_button.click(
        clear_fields,
        outputs=[image_input, url_input, image_output_pretrained, image_output_finetuned, count_pretrained_text, count_finetuned_text]
    )
iface.launch()