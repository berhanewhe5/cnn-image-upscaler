import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

from src.model import PixelShuffle

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# cache models so they only load once
MODEL_CACHE = {}

ALLOWED_SCALES = {2, 3, 4, 8}


def get_model(scale: int):
    if scale not in ALLOWED_SCALES:
        raise ValueError(f"Unsupported scale: {scale}")

    if scale not in MODEL_CACHE:
        model_path = CHECKPOINT_DIR / f"espcn_x{scale}_best.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")

        MODEL_CACHE[scale] = tf.keras.models.load_model(
            model_path,
            custom_objects={"PixelShuffle": PixelShuffle},
            compile=False,
        )
    return MODEL_CACHE[scale]


def upscale_image(input_path: Path, scale: int) -> str:
    model = get_model(scale)

    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    x = arr[np.newaxis, ...]

    pred = model.predict(x, verbose=0)[0]
    pred = np.clip(pred, 0.0, 1.0)

    out = Image.fromarray((pred * 255).astype(np.uint8))
    output_name = f"{input_path.stem}_x{scale}_sr.png"
    output_path = UPLOAD_DIR / output_name
    out.save(output_path)

    return output_name


@app.route("/", methods=["GET", "POST"])
def index():
    output_file = None
    original_file = None
    selected_scale = 2
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        scale_str = request.form.get("scale", "2")

        try:
            selected_scale = int(scale_str)
            if selected_scale not in ALLOWED_SCALES:
                raise ValueError("Invalid scale selected.")

            if not file or file.filename == "":
                raise ValueError("Please upload an image.")

            original_name = Path(file.filename).name
            original_path = UPLOAD_DIR / original_name
            file.save(original_path)

            output_file = upscale_image(original_path, selected_scale)
            original_file = original_name

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        output_file=output_file,
        original_file=original_file,
        selected_scale=selected_scale,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
