import io
import os
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
import onnxruntime as ort
from torchvision import datasets

MODEL_NAME = "mobilenetv3_skin"
DATA_DIR = "train"
MODEL_PATH = f"export/{MODEL_NAME}.onnx"

# AnythingLLM configuration (set these as env vars)
ANYL_BASE_URL = os.environ.get("ANYL_BASE_URL", "http://127.0.0.1:3001")
ANYL_API_KEY = os.environ.get("ANYL_API_KEY", "")
ANYL_WORKSPACE = os.environ.get("ANYL_WORKSPACE", "your-workspace-slug")

# Load class names from your train folders
base_ds = datasets.ImageFolder(DATA_DIR)
classes = base_ds.classes

# Load ONNX model once
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
    # Build float32 tensor expected by ONNX Runtime
    x = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return x.astype(np.float32)[None, ...]  # (1, 3, 224, 224)


app = Flask(__name__)
# Allow all origins in dev so Vite (5173/5174/etc.) can call the API
CORS(app)


HTML = """
<!doctype html>
<title>Skin Classifier</title>
<h1>Upload a skin image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image accept="image/*">
  <input type=submit value=Predict>
</form>
{% if result %}
  <h2>Prediction:</h2>
  <p>Class: {{ result.class_name }} (confidence {{ '{:.3f}'.format(result.confidence) }})</p>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    """Simple HTML form for manual testing."""
    result = None
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        x = preprocess_image(file.read())
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: x})[0]
        probs = np.exp(outputs) / np.exp(outputs).sum(axis=1, keepdims=True)
        top_idx = int(probs.argmax(axis=1)[0])
        top_prob = float(probs[0, top_idx])
        result = {"class_name": classes[top_idx], "confidence": top_prob}
    return render_template_string(HTML, result=result)


@app.route("/classify_image", methods=["POST"])
def classify_image():
    """
    API endpoint used by the React frontend.
    Expects multipart/form-data with field name 'image'.
    Returns JSON: { "result": <class name>, "confidence": <float> }.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    x = preprocess_image(file.read())
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})[0]
    probs = np.exp(outputs) / np.exp(outputs).sum(axis=1, keepdims=True)
    top_idx = int(probs.argmax(axis=1)[0])
    top_prob = float(probs[0, top_idx])

    return jsonify({"result": classes[top_idx], "confidence": top_prob})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Proxy endpoint for AnythingLLM so the API key stays on the backend.
    Expects JSON: { "message": "<user text>" }
    Returns JSON: { "reply": "<llm reply>", "raw": <full AnythingLLM payload> }
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    if not ANYL_API_KEY:
        return jsonify({"error": "ANYL_API_KEY is not set on the backend"}), 500

    url = f"{ANYL_BASE_URL}/api/v1/workspace/{ANYL_WORKSPACE}/chat"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANYL_API_KEY,
    }

    try:
        resp = requests.post(url, headers=headers, json={"message": message}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": "Failed to call AnythingLLM", "detail": str(e)}), 502

    payload = resp.json()
    # Try a few common keys for the main text; fall back to the whole payload.
    reply = payload.get("response") or payload.get("text") or payload.get("message") or ""
    return jsonify({"reply": reply, "raw": payload})


if __name__ == "__main__":
    # Use a port that isn't taken by macOS services like AirPlay (e.g. 8001)
    app.run(host="0.0.0.0", port=8001, debug=True)
