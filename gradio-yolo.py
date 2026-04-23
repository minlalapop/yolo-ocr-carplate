from pathlib import Path
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from rapidocr import RapidOCR

# =========================
# Config
# =========================
MODEL_PATH = "/Users/yasmine/Documents/yolo copy/runs/detect/train-3/weights/best.pt"
model = YOLO(MODEL_PATH)
engine = RapidOCR()


# =========================
# SAME JUPYTER FUNCTIONS
# =========================
def extract_plate(img_org, bounding_box, save_path=None, show=True):
    x1, y1, x2, y2 = map(int, bounding_box)

    w, h = img_org.size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    plate = img_org.crop((x1, y1, x2, y2))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plate.save(save_path)

    return plate


def preprocess_plate_for_ocr(img_path, save_path=None, show=True):
    img = Image.open(img_path).convert("L")

    w, h = img.size
    img = img.resize((w * 3, h * 3), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = img.filter(ImageFilter.SHARPEN)

    arr = np.array(img)
    arr = cv2.adaptiveThreshold(
        arr,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    out = Image.fromarray(arr)

    if save_path:
        out.save(save_path)

    return out


def extract_text(save_path):
    result = engine(str(save_path))
    return result


def crop_image(img_path, number):
    img = Image.open(img_path)
    results = model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None

    save_path = f"extracted_plates/Cars{number}_plate.png"
    extract_plate(
        img_org=img,
        bounding_box=boxes[0],
        save_path=save_path,
        show=False,
    )

    return save_path


def analyse_image(number):
    img_path = f"datasets/License-Plate-Data/test/images/Cars{number}.png"
    save_path = f"extracted_plates/Cars{number}_plate_processed.png"

    extract_path = crop_image(img_path, number)
    if extract_path is None:
        return None

    preprocess_plate_for_ocr(extract_path, save_path, show=False)
    result = extract_text(save_path)

    if result is None or result.txts is None:
        return []

    return result.txts


# =========================
# SIMPLE HELPERS FOR GRADIO
# =========================
def save_uploaded_image(image):
    temp_dir = Path(tempfile.gettempdir()) / "gradio_plate_app"
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / "uploaded_input.png"
    image.save(input_path)
    return input_path


def yolo_annotated_image(img_path):
    results = model(str(img_path))
    plotted = results[0].plot()
    return Image.fromarray(plotted)


def pipeline(image):
    if image is None:
        return None, None, "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    input_path = save_uploaded_image(image)

    # YOLO image with bbox
    yolo_img = yolo_annotated_image(input_path)

    # Use SAME notebook logic, with fixed temp number
    number = 999999
    raw_crop_path = f"extracted_plates/Cars{number}_plate.png"
    processed_crop_path = f"extracted_plates/Cars{number}_plate_processed.png"

    # crop using same logic
    img = Image.open(input_path)
    results = model(str(input_path))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return yolo_img, None, "No plate detected"

    extract_plate(
        img_org=img,
        bounding_box=boxes[0],
        save_path=raw_crop_path,
        show=False,
    )

    processed_plate = preprocess_plate_for_ocr(raw_crop_path, processed_crop_path, show=False)
    result = extract_text(processed_crop_path)

    if result is None or result.txts is None:
        text = "No text detected"
    else:
        text = " ".join([str(t).strip() for t in result.txts if str(t).strip()])
        if not text:
            text = "No text detected"

    return yolo_img, processed_plate, text


# =========================
# SIMPLE PINK UI
# =========================
CSS = """
body { background: #fff7fb; }
.gradio-container { background: #fff7fb; font-family: 'Segoe UI', sans-serif; }
button {
    background: #f8d8e6 !important;
    color: #7b5a67 !important;
    border: 1px solid #efc6d6 !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style='text-align:center;'>
            <h1 style='color:#c77a98;'>Plate OCR</h1>
            <p style='color:#9a7584;'>Upload image -> YOLO detect -> crop plate -> OCR</p>
        </div>
        """
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload image")
        detected_image = gr.Image(type="pil", label="YOLO detection")

    with gr.Row():
        cropped_plate = gr.Image(type="pil", label="Processed plate")
        ocr_text = gr.Textbox(label="OCR result", lines=3)

    run_btn = gr.Button("Run")
    clear_btn = gr.Button("Clear")

    run_btn.click(
        fn=pipeline,
        inputs=input_image,
        outputs=[detected_image, cropped_plate, ocr_text],
    )

    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=None,
        outputs=[detected_image, cropped_plate, ocr_text],
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
