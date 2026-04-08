import io
import json
import os
from html import escape

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps

app = Flask(
    __name__,
    static_folder=os.path.dirname(os.path.abspath(__file__)),
    static_url_path="",
)
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "1")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_INPUT_VALUE_RANGE = "normalized_0_1"
TOP_K_PREDICTIONS = 8
LEGACY_APPLE_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

# ── Runtime state ────────────────────────────────────────────────────────────
model = None
model_signature = None
input_key = None
output_key = None
class_names = []
image_size = DEFAULT_IMAGE_SIZE
input_value_range = DEFAULT_INPUT_VALUE_RANGE


def load_json(path: str):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def infer_image_size_from_signature(signature) -> int:
    if signature is None:
        return DEFAULT_IMAGE_SIZE

    spec = signature.structured_input_signature[1][input_key]
    shape = spec.shape

    if len(shape) >= 4 and shape[1] and shape[2]:
        return int(shape[1])

    return DEFAULT_IMAGE_SIZE


def output_dim_from_signature(signature):
    if signature is None:
        return None

    out_spec = signature.structured_outputs[output_key]
    if len(out_spec.shape) >= 2 and out_spec.shape[-1]:
        return int(out_spec.shape[-1])

    return None


def generated_class_names(size: int) -> list[str]:
    return [f"class_{idx}" for idx in range(size)]


def infer_input_value_range(metadata: dict) -> str:
    explicit = metadata.get("input_value_range")
    if explicit in {"normalized_0_1", "raw_0_255"}:
        return explicit

    if metadata.get("model_name") == "leafscan_plantvillage_transfer":
        return "raw_0_255"

    return DEFAULT_INPUT_VALUE_RANGE


def load_model():
    global model, model_signature, input_key, output_key, class_names, image_size, input_value_range

    saved_model_file = os.path.join(MODEL_PATH, "saved_model.pb")
    if not os.path.exists(saved_model_file):
        print(f"[WARNING] saved_model.pb not found at: {saved_model_file}")
        return

    try:
        model = tf.saved_model.load(MODEL_PATH)
        model_signature = model.signatures["serving_default"]
        input_key = list(model_signature.structured_input_signature[1].keys())[0]
        output_key = list(model_signature.structured_outputs.keys())[0]

        metadata = load_json(os.path.join(MODEL_PATH, "model_metadata.json")) or {}
        labels_payload = load_json(os.path.join(MODEL_PATH, "class_names.json"))

        loaded_class_names = metadata.get("class_names")
        if loaded_class_names is None and isinstance(labels_payload, dict):
            loaded_class_names = labels_payload.get("class_names")
        if loaded_class_names is None and isinstance(labels_payload, list):
            loaded_class_names = labels_payload

        output_dim = output_dim_from_signature(model_signature)
        if not loaded_class_names and output_dim == len(LEGACY_APPLE_CLASS_NAMES):
            loaded_class_names = LEGACY_APPLE_CLASS_NAMES
        if not loaded_class_names and output_dim:
            loaded_class_names = generated_class_names(output_dim)

        class_names = loaded_class_names or []
        image_size = int(metadata.get("image_size") or infer_image_size_from_signature(model_signature))
        input_value_range = infer_input_value_range(metadata)

        print(
            "[INFO] TensorFlow model loaded successfully.",
            f"classes={len(class_names)}",
            f"image_size={image_size}",
            f"input_value_range={input_value_range}",
        )
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")


load_model()


# ── Helpers ──────────────────────────────────────────────────────────────────
def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()


def format_label(value: str) -> str:
    return (
        value.replace("___", " - ")
        .replace("_", " ")
        .replace(" ,", ",")
        .replace("( ", "(")
        .replace(" )", ")")
        .strip()
    )


def get_plant_name(class_name: str) -> str:
    plant_name = class_name.split("___")[0]
    return format_label(plant_name)


def get_disease_name(class_name: str) -> str:
    disease_name = class_name.split("___", 1)[1] if "___" in class_name else class_name
    return format_label(disease_name)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(
        img,
        (image_size, image_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )

    arr = np.array(img, dtype=np.float32)
    if input_value_range == "normalized_0_1":
        arr /= 255.0
    return np.expand_dims(arr, axis=0)


def ensure_probabilities(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    row = preds[0]
    if np.any(row < 0) or np.any(row > 1) or not np.isclose(np.sum(row), 1.0, atol=1e-3):
        row = tf.nn.softmax(row).numpy()
        preds[0] = row

    return preds


def run_model(img_array: np.ndarray) -> np.ndarray:
    preds = model_signature(**{input_key: tf.constant(img_array)})[output_key].numpy()
    return ensure_probabilities(preds)


def build_probability_rows(preds: np.ndarray) -> list[dict]:
    probs = preds[0]
    rows = []

    for idx, probability in enumerate(probs):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        rows.append(
            {
                "class": name,
                "display_name": format_label(name),
                "plant_name": get_plant_name(name),
                "disease_name": get_disease_name(name),
                "probability": round(float(probability) * 100, 2),
                "healthy": is_healthy(name),
            }
        )

    rows.sort(key=lambda item: item["probability"], reverse=True)
    return rows


def build_ai_html(predicted_class: str, confidence: float) -> str:
    plant_name = get_plant_name(predicted_class)
    disease_name = get_disease_name(predicted_class)
    healthy = is_healthy(predicted_class)

    if healthy:
        summary = "The uploaded leaf most closely matches a healthy sample from the PlantVillage dataset."
        next_step = "Keep monitoring the plant, and retake photos in natural light if you notice any new spots, rust, or curling."
    elif confidence >= 85:
        summary = "The model found a strong match for this disease pattern in the PlantVillage dataset."
        next_step = "Compare the leaf with a few nearby leaves and isolate affected plants if the same symptoms are spreading."
    else:
        summary = "The model found a likely match, but the confidence is moderate rather than decisive."
        next_step = "Retake the photo with a single leaf, plain background, and better lighting to confirm the diagnosis."

    return f"""
<p><strong>Plant:</strong> {escape(plant_name)}</p>
<p><strong>Detected condition:</strong> {escape(disease_name)}</p>
<p><strong>Status:</strong> {"Healthy" if healthy else "Disease symptoms detected"}</p>
<p><strong>Confidence:</strong> {confidence:.2f}%</p>
<p>{escape(summary)}</p>
<p>{escape(next_step)}</p>
""".strip()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "num_classes": len(class_names),
            "image_size": image_size,
            "model_path": MODEL_PATH,
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    if model is None or model_signature is None:
        return jsonify({"error": "TensorFlow model not loaded"}), 500

    file = request.files["image"]
    file_bytes = file.read()

    try:
        img_array = preprocess_image(file_bytes)
        preds = run_model(img_array)
        all_probs = build_probability_rows(preds)

        top_prediction = all_probs[0]
        predicted_class = top_prediction["class"]
        confidence = top_prediction["probability"]

        model_result = {
            "predicted_class": predicted_class,
            "display_name": top_prediction["display_name"],
            "plant_name": top_prediction["plant_name"],
            "disease_name": top_prediction["disease_name"],
            "confidence": confidence,
            "is_healthy": top_prediction["healthy"],
            "all_probabilities": all_probs[:TOP_K_PREDICTIONS],
        }

        return jsonify(
            {
                "model_result": model_result,
                "ai_html": build_ai_html(predicted_class, confidence),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("[INFO] Starting LeafScan backend on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
