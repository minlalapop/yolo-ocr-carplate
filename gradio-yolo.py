from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# RapidOCR import compatibility:
# - Older/common package: rapidocr_onnxruntime
# - Newer package name in the RapidOCR project ecosystem: rapidocr
try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:
    try:
        from rapidocr import RapidOCR  # type: ignore
    except Exception as e:
        raise ImportError(
            "RapidOCR is not installed. Install one of: `pip install rapidocr_onnxruntime` "
            "or `pip install rapidocr`."
        ) from e


# ---------- Configuration ----------
MODEL_PATH = Path("runs/detect/train-3/weights/best.pt")
CLASS_NAME = "license_plate"
CONF_THRESHOLD = 0.25


# ---------- Model loading ----------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO weights not found at: {MODEL_PATH.resolve()}\n"
        "Update MODEL_PATH at the top of gradio_plate_ocr_app.py if needed."
    )

model = YOLO(str(MODEL_PATH))
ocr_engine = RapidOCR()


# ---------- Core helpers ----------
def extract_plate(
    img_org: Image.Image,
    bounding_box: Tuple[float, float, float, float],
    save_path: Optional[str] = None,
) -> Image.Image:
    x1, y1, x2, y2 = map(int, bounding_box)

    w, h = img_org.size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    plate = img_org.crop((x1, y1, x2, y2))

    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plate.save(save_path)

    return plate


def draw_prediction(image: Image.Image, bbox: Tuple[float, float, float, float], score: float) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = map(int, bbox)

    draw.rounded_rectangle((x1, y1, x2, y2), outline=(255, 20, 147), width=5, radius=12)
    label = f"{CLASS_NAME} {score:.2f}"
    text_x = x1
    text_y = max(0, y1 - 24)
    draw.rounded_rectangle((text_x, text_y, text_x + 180, text_y + 22), fill=(255, 182, 193))
    draw.text((text_x + 8, text_y + 3), label, fill=(90, 0, 60))
    return annotated


def run_ocr(plate_img: Image.Image) -> str:
    plate_np = np.array(plate_img)
    output = ocr_engine(plate_np)

    # Newer RapidOCR objects may return a RapidOCROutput object
    if hasattr(output, "txts"):
        texts = [str(t).strip() for t in getattr(output, "txts", []) if str(t).strip()]
        return " ".join(texts) if texts else "No text detected."

    # Older variants may return a tuple/list structure
    if isinstance(output, tuple) and len(output) >= 1:
        result = output[0]
        if not result:
            return "No text detected."
        texts = []
        for item in result:
            if len(item) >= 2:
                texts.append(str(item[1]).strip())
        return " ".join(t for t in texts if t) if texts else "No text detected."

    if isinstance(output, list):
        if not output:
            return "No text detected."
        texts = []
        for item in output:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                texts.append(str(item[1]).strip())
        return " ".join(t for t in texts if t) if texts else "No text detected."

    return "No text detected."


def detect_plate_and_ocr(image: Image.Image):
    if image is None:
        return None, None, "Please upload an image.", None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return image, None, "No license plate detected.", None

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Keep the highest-confidence detection
    best_idx = int(np.argmax(confs))
    best_box = tuple(xyxy[best_idx].tolist())
    best_conf = float(confs[best_idx])

    annotated = draw_prediction(image, best_box, best_conf)

    temp_dir = Path(tempfile.gettempdir()) / "girly_plate_crops"
    crop_path = temp_dir / "plate_crop.png"
    plate_crop = extract_plate(image, best_box, save_path=str(crop_path))

    ocr_text = run_ocr(plate_crop)

    return annotated, plate_crop, ocr_text, str(crop_path)


# ---------- UI ----------
PINK_CSS = """
body {
    background: #fff8fc;
}
.gradio-container {
    font-family: 'Segoe UI', 'Trebuchet MS', sans-serif;
    background: #fff8fc;
}
#main-card {
    border: 1px solid #f5d3e0;
    border-radius: 22px;
    background: #fffdff;
    box-shadow: 0 8px 24px rgba(230, 177, 199, 0.12);
    padding: 16px;
}
.pink-title {
    text-align: center;
    color: #c76a93;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.pink-subtitle {
    text-align: center;
    color: #9d7282;
    margin-bottom: 1rem;
}
button {
    background: #f8d7e4 !important;
    color: #7b4e61 !important;
    border: 1px solid #efc4d5 !important;
    border-radius: 14px !important;
}
button:hover {
    background: #f4cfde !important;
}
"""

with gr.Blocks(title="Plate detector : YOLO et OCR") as demo:
    gr.HTML("<div class='pink-title'>Plate detector : YOLO et OCR</div>")
    gr.HTML("<div class='pink-subtitle'>Uplodez une image de voiture, detecter la plaque avec YOLO et extractez le texte avec OCR</div>")

    with gr.Column(elem_id="main-card"):
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload your image")
            annotated_output = gr.Image(type="pil", label="YOLO detection")

        with gr.Row():
            crop_output = gr.Image(type="pil", label="Extracted plate")
            with gr.Column():
                text_output = gr.Textbox(label="OCR result", lines=3, placeholder="Detected text will appear here...")
                file_output = gr.File(label="Saved cropped plate")

        with gr.Row():
            run_btn = gr.Button("Detect plate + OCR")
            clear_btn = gr.Button("Clear")

    run_btn.click(
        fn=detect_plate_and_ocr,
        inputs=input_image,
        outputs=[annotated_output, crop_output, text_output, file_output],
    )

    clear_btn.click(
        fn=lambda: (None, None, "", None),
        inputs=None,
        outputs=[annotated_output, crop_output, text_output, file_output],
    )

if __name__ == "__main__":
    demo.launch()
