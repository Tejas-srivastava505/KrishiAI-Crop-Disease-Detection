# =============================================================================
# app.py  —  KrishiAI Flask Backend
#
# Serves the farmer mobile web app AND the prediction API.
# The TFLite model runs entirely on the server, so the farmer's ₹8k phone
# only needs to send a photo — no ML library needed on-device.
#
# Routes:
#   GET  /                    → Farmer mobile web app (farmer.html)
#   POST /predict             → Accept image, return JSON diagnosis
#   GET  /history             → Last 50 scans for this farmer (session-based)
#   GET  /health              → Server health check
#
# Run:
#   pip install flask flask-cors pillow
#   python app.py
#
# Then open http://localhost:5000 on any phone on the same WiFi.
# For public access: use ngrok → ngrok http 5000
# =============================================================================

import os
import sys
import json
import uuid
import time
import base64
import logging
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS

# Add project root so we can import config + inference engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TFLITE_DIR, TFLITE_MODEL_NAME, IMAGE_SIZE, NUM_CLASSES,
    CLASS_NAMES, KANNADA_ALERTS, TREATMENT_MAP, LOGS_DIR
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "server.log"), encoding="utf-8"),
    ]
)
log = logging.getLogger("KrishiAI")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "krishiai-udupi-2025"   # change in production
CORS(app)

# Scan history stored in memory (use SQLite/PostgreSQL for production)
SCAN_HISTORY = []
MAX_HISTORY  = 200

# ── Load TFLite inference engine once at startup ───────────────────────────────
try:
    from step4_export_and_infer import KrishiAIInference
    ENGINE = KrishiAIInference()
    MODEL_READY = True
    log.info("✅ TFLite model loaded successfully")
except Exception as e:
    log.warning(f"⚠️  TFLite model not found ({e}). Running in DEMO mode.")
    ENGINE      = None
    MODEL_READY = False


# ──────────────────────────────────────────────────────────────────────────────
# MOCK ENGINE for demo / before training is complete
# ──────────────────────────────────────────────────────────────────────────────

DEMO_RESULTS = [
    {
        "disease": "Rice Blast",
        "confidence": 0.93,
        "confidence_pct": "93.0%",
        "risk_level": "HIGH",
        "kannada_alert": "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ ಪತ್ತೆ. ಟ್ರೈಸೈಕ್ಲಾಜ಼ೋಲ್ ತಕ್ಷಣ ಸಿಂಪಡಿಸಿ.",
        "treatment": ["Tricyclazole 75WP @ 0.6 g/L", "Isoprothiolane 40EC @ 1.5 mL/L", "Avoid excess nitrogen", "Drain stagnant water"],
        "top_k": [
            {"disease": "Rice Blast",         "confidence": 0.93},
            {"disease": "Brown Spot",         "confidence": 0.05},
            {"disease": "Bacterial Blight",   "confidence": 0.02},
        ],
        "inference_ms": 312.0,
    },
    {
        "disease": "Healthy Rice",
        "confidence": 0.96,
        "confidence_pct": "96.0%",
        "risk_level": "LOW",
        "kannada_alert": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
        "treatment": ["Continue regular monitoring", "Preventive fungicide (optional)"],
        "top_k": [
            {"disease": "Healthy Rice",       "confidence": 0.96},
            {"disease": "Rice Blast",         "confidence": 0.03},
            {"disease": "Brown Spot",         "confidence": 0.01},
        ],
        "inference_ms": 287.0,
    },
    {
        "disease": "Bacterial Blight",
        "confidence": 0.87,
        "confidence_pct": "87.0%",
        "risk_level": "MODERATE",
        "kannada_alert": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಒಣಗುವಿಕೆ. ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಬಳಸಿ.",
        "treatment": ["Copper Oxychloride 50WP @ 3 g/L", "Streptocycline @ 0.5 g/L", "Avoid flood irrigation"],
        "top_k": [
            {"disease": "Bacterial Blight",   "confidence": 0.87},
            {"disease": "Healthy Rice",       "confidence": 0.08},
            {"disease": "Rice Blast",         "confidence": 0.05},
        ],
        "inference_ms": 298.0,
    },
]
_demo_counter = 0


def run_inference(pil_image: Image.Image) -> dict:
    """Run real or demo inference on a PIL image."""
    global _demo_counter

    # Resize image to model input size
    pil_image = ImageOps.fit(pil_image.convert("RGB"), IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(pil_image)   # (224, 224, 3) uint8

    if MODEL_READY and ENGINE is not None:
        result = ENGINE.predict(img_array)
    else:
        # Demo mode — cycle through sample results
        result = DEMO_RESULTS[_demo_counter % len(DEMO_RESULTS)].copy()
        _demo_counter += 1
        result["inference_ms"] = round(200 + np.random.uniform(0, 200), 1)
        result["demo_mode"] = True

    return result


# ──────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY,
        "mode":        "live" if MODEL_READY else "demo",
        "timestamp":   datetime.now().isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a crop photo and return disease diagnosis.

    Accepts:
      • multipart/form-data  with field "image"
      • application/json     with field "image_b64" (base64-encoded JPEG/PNG)

    Returns JSON:
      {
        "scan_id":        "abc123",
        "disease":        "Rice Blast",
        "confidence_pct": "93.0%",
        "risk_level":     "HIGH",
        "kannada_alert":  "ಭತ್ತದ ...",
        "treatment":      ["Tricyclazole 75WP ...", ...],
        "top_k":          [...],
        "inference_ms":   312.0,
        "timestamp":      "2026-03-22T08:14:00",
        "demo_mode":      false,
      }
    """
    t_start = time.perf_counter()

    # ── Parse image ────────────────────────────────────────────────────────────
    pil_image = None

    if "image" in request.files:
        # multipart upload (standard HTML form or camera capture)
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        try:
            pil_image = Image.open(file.stream)
        except Exception as e:
            return jsonify({"error": f"Cannot read image: {e}"}), 400

    elif request.is_json and "image_b64" in request.json:
        # Base64 upload (from JS fetch with canvas snapshot)
        try:
            b64_data = request.json["image_b64"]
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]   # strip data:image/jpeg;base64,
            img_bytes = base64.b64decode(b64_data)
            pil_image = Image.open(BytesIO(img_bytes))
        except Exception as e:
            return jsonify({"error": f"Cannot decode base64 image: {e}"}), 400
    else:
        return jsonify({"error": "Send image as multipart 'image' field or JSON 'image_b64'"}), 400

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        result = run_inference(pil_image)
    except Exception as e:
        log.error(f"Inference error: {e}")
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # ── Build response ─────────────────────────────────────────────────────────
    scan_id   = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Farmer name from form (optional)
    farmer_name = request.form.get("farmer_name", "") or (
        request.json.get("farmer_name", "") if request.is_json else ""
    )
    crop_type   = request.form.get("crop_type", "paddy") or (
        request.json.get("crop_type", "paddy") if request.is_json else "paddy"
    )

    response = {
        "scan_id":        scan_id,
        "timestamp":      timestamp,
        "farmer_name":    farmer_name,
        "crop_type":      crop_type,
        **result,
        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
    }

    # ── Store in history ───────────────────────────────────────────────────────
    SCAN_HISTORY.insert(0, {
        "scan_id":       scan_id,
        "timestamp":     timestamp,
        "farmer_name":   farmer_name,
        "crop_type":     crop_type,
        "disease":       result["disease"],
        "confidence_pct": result["confidence_pct"],
        "risk_level":    result["risk_level"],
    })
    if len(SCAN_HISTORY) > MAX_HISTORY:
        SCAN_HISTORY.pop()

    log.info(f"Scan {scan_id} | {result['disease']} | {result['confidence_pct']} | {farmer_name or 'anon'}")

    return jsonify(response)


@app.route("/history")
def history():
    """Return the last 50 scans."""
    return jsonify(SCAN_HISTORY[:50])


@app.route("/")
def farmer_app():
    """Serve the farmer mobile web app."""
    return render_template_string(FARMER_HTML)


# ──────────────────────────────────────────────────────────────────────────────
# FARMER MOBILE WEB APP (inline HTML — no separate file needed)
# Designed for low-end Android phones, bad network, one-handed use.
# ──────────────────────────────────────────────────────────────────────────────

FARMER_HTML = r"""<!DOCTYPE html>
<html lang="kn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#1a3d1a">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>KrishiAI — ಬೆಳೆ ರೋಗ ಪತ್ತೆ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;500;600;700;800&family=Noto+Sans+Kannada:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
  --green-deep:   #0d2b0d;
  --green-dark:   #1a3d1a;
  --green-mid:    #2d6a2d;
  --green-main:   #3d8b3d;
  --green-light:  #5aaf5a;
  --green-pale:   #c8e6c9;
  --gold:         #f4a800;
  --gold-light:   #fff3cd;
  --red:          #c62828;
  --red-light:    #ffebee;
  --orange:       #e65100;
  --orange-light: #fff3e0;
  --text-dark:    #1a2e1a;
  --text-mid:     #3d5a3d;
  --text-light:   #6b8f6b;
  --white:        #ffffff;
  --surface:      #f4faf4;
  --border:       #c3e6c3;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 16px; -webkit-text-size-adjust: 100%; }

body {
  font-family: 'Baloo 2', 'Noto Sans Kannada', sans-serif;
  background: var(--green-deep);
  color: var(--text-dark);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── TOP BAR ── */
.topbar {
  background: var(--green-dark);
  padding: 14px 20px 12px;
  display: flex; align-items: center; gap: 12px;
  position: sticky; top: 0; z-index: 100;
  border-bottom: 2px solid var(--green-mid);
}
.topbar-logo { font-size: 26px; line-height: 1; }
.topbar-name { font-size: 20px; font-weight: 800; color: #7ed87e; letter-spacing: -0.3px; }
.topbar-sub  { font-size: 11px; color: #5a8a5a; font-family: 'Noto Sans Kannada', sans-serif; }
.topbar-right { margin-left: auto; }
.mode-pill {
  font-size: 10px; font-weight: 700; padding: 3px 10px;
  border-radius: 20px; background: rgba(90,175,90,0.2);
  color: #7ed87e; border: 1px solid rgba(90,175,90,0.3);
  letter-spacing: 0.5px;
}

/* ── SCROLL CONTAINER ── */
.scroll-wrap {
  background: var(--surface);
  min-height: calc(100vh - 58px);
  padding-bottom: 40px;
}

/* ── HERO BAND ── */
.hero {
  background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 100%);
  padding: 24px 20px 28px;
  text-align: center;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-title {
  font-size: 22px; font-weight: 800; color: white;
  line-height: 1.2; margin-bottom: 4px; position: relative;
}
.hero-kannada {
  font-family: 'Noto Sans Kannada', sans-serif;
  font-size: 15px; color: rgba(255,255,255,0.75);
  margin-bottom: 16px; position: relative;
}
.hero-steps {
  display: flex; justify-content: center; gap: 6px;
  position: relative;
}
.hero-step {
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 20px; padding: 5px 12px;
  font-size: 11px; color: rgba(255,255,255,0.85);
  display: flex; align-items: center; gap: 4px;
}

/* ── MAIN CARD ── */
.card {
  background: var(--white);
  margin: 16px;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  border: 1px solid var(--border);
}
.card-header {
  background: var(--green-dark);
  padding: 14px 18px;
  display: flex; align-items: center; gap: 10px;
}
.card-header-icon { font-size: 22px; }
.card-header-title { font-size: 16px; font-weight: 700; color: white; }
.card-header-sub { font-size: 11px; color: rgba(255,255,255,0.6); font-family: 'Noto Sans Kannada'; }
.card-body { padding: 18px; }

/* ── CROP SELECTOR ── */
.crop-grid {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 8px; margin-bottom: 18px;
}
.crop-btn {
  border: 2px solid var(--border);
  border-radius: 14px; padding: 12px 6px;
  background: var(--surface); cursor: pointer;
  text-align: center; transition: all 0.18s;
  font-family: 'Baloo 2', 'Noto Sans Kannada', sans-serif;
}
.crop-btn.active {
  border-color: var(--green-main);
  background: var(--green-pale);
}
.crop-btn-icon { font-size: 26px; display: block; margin-bottom: 3px; }
.crop-btn-name { font-size: 12px; font-weight: 600; color: var(--text-dark); }
.crop-btn-kannada { font-size: 10px; color: var(--text-light); font-family: 'Noto Sans Kannada'; }

/* ── UPLOAD ZONE ── */
.upload-zone {
  border: 2.5px dashed var(--green-light);
  border-radius: 16px; min-height: 200px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; cursor: pointer;
  background: linear-gradient(135deg, #f0fbf0, #e8f5e8);
  position: relative; overflow: hidden;
  transition: all 0.2s;
  -webkit-tap-highlight-color: transparent;
}
.upload-zone.has-image { min-height: 240px; border-style: solid; border-color: var(--green-main); }
.upload-zone:active { background: #e0f5e0; transform: scale(0.99); }
.upload-zone input[type="file"] {
  position: absolute; inset: 0;
  opacity: 0; cursor: pointer; font-size: 0;
}
.upload-icon { font-size: 48px; pointer-events: none; }
.upload-text { font-size: 16px; font-weight: 700; color: var(--green-mid); pointer-events: none; }
.upload-hint { font-size: 12px; color: var(--text-light); font-family: 'Noto Sans Kannada'; pointer-events: none; }
.preview-img {
  width: 100%; height: 240px;
  object-fit: cover; border-radius: 14px;
  display: none;
}
.preview-overlay {
  position: absolute; bottom: 8px; right: 8px;
  background: rgba(0,0,0,0.55); color: white;
  font-size: 11px; padding: 4px 10px; border-radius: 10px;
  display: none;
}

/* ── CAMERA BUTTONS ── */
.capture-row {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px; margin-top: 14px;
}
.btn-camera {
  padding: 14px;
  border: none; border-radius: 14px;
  font-family: 'Baloo 2', sans-serif;
  font-size: 14px; font-weight: 700;
  cursor: pointer; transition: all 0.18s;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  -webkit-tap-highlight-color: transparent;
}
.btn-camera:active { transform: scale(0.97); }
.btn-camera.primary {
  background: var(--green-main);
  color: white;
  box-shadow: 0 4px 12px rgba(61,139,61,0.35);
}
.btn-camera.secondary {
  background: var(--surface);
  color: var(--green-mid);
  border: 2px solid var(--border);
}
.btn-camera .cam-icon { font-size: 20px; }

/* ── FARMER INFO ── */
.input-row { margin-bottom: 12px; }
.input-label {
  font-size: 12px; font-weight: 600; color: var(--text-mid);
  margin-bottom: 4px; display: block;
}
.input-label .label-kn { font-family: 'Noto Sans Kannada'; font-weight: 400; color: var(--text-light); font-size: 11px; margin-left: 4px; }
.input-field {
  width: 100%; padding: 12px 14px;
  border: 2px solid var(--border); border-radius: 12px;
  font-family: 'Baloo 2', 'Noto Sans Kannada', sans-serif;
  font-size: 15px; color: var(--text-dark);
  background: var(--surface); outline: none;
  transition: border-color 0.2s;
  -webkit-appearance: none;
}
.input-field:focus { border-color: var(--green-main); background: white; }

/* ── ANALYZE BUTTON ── */
.btn-analyze {
  width: 100%; padding: 18px;
  background: linear-gradient(135deg, #3d8b3d, #2d6a2d);
  border: none; border-radius: 16px;
  font-family: 'Baloo 2', sans-serif;
  font-size: 18px; font-weight: 800;
  color: white; cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 6px 20px rgba(45,106,45,0.4);
  display: flex; align-items: center; justify-content: center; gap: 8px;
  -webkit-tap-highlight-color: transparent;
  margin-top: 4px;
}
.btn-analyze:active { transform: scale(0.98); box-shadow: 0 2px 8px rgba(45,106,45,0.3); }
.btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-analyze .kn { font-family: 'Noto Sans Kannada'; font-size: 14px; font-weight: 600; opacity: 0.85; }

/* ── LOADING ── */
.loading-card {
  display: none;
  margin: 0 16px 16px;
  background: var(--green-dark);
  border-radius: 20px; padding: 28px 20px;
  text-align: center;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.loading-spinner {
  width: 48px; height: 48px;
  border: 4px solid rgba(255,255,255,0.2);
  border-top-color: #7ed87e;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 14px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-size: 16px; font-weight: 700; color: white; margin-bottom: 4px; }
.loading-sub  { font-size: 12px; color: rgba(255,255,255,0.55); font-family: 'Noto Sans Kannada'; }

/* ── RESULT CARD ── */
.result-card {
  display: none;
  margin: 0 16px 16px;
  border-radius: 20px; overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  animation: slideUp 0.4s cubic-bezier(.16,1,.3,1);
}
@keyframes slideUp { from { opacity:0; transform:translateY(20px) } to { opacity:1; transform:none } }

.result-header {
  padding: 20px 18px 16px;
  display: flex; align-items: flex-start; gap: 14px;
}
.result-header.danger  { background: var(--red-light);    border-bottom: 3px solid var(--red); }
.result-header.warning { background: var(--orange-light); border-bottom: 3px solid var(--orange); }
.result-header.safe    { background: var(--green-pale);   border-bottom: 3px solid var(--green-main); }
.result-icon { font-size: 40px; line-height: 1; flex-shrink: 0; }
.result-disease { font-size: 20px; font-weight: 800; line-height: 1.2; }
.result-header.danger  .result-disease { color: var(--red); }
.result-header.warning .result-disease { color: var(--orange); }
.result-header.safe    .result-disease { color: #1b5e20; }
.result-confidence { font-size: 13px; font-weight: 600; color: var(--text-mid); margin-top: 3px; }
.result-scan-id { font-size: 10px; color: var(--text-light); font-family: monospace; margin-top: 2px; }

.conf-bar {
  height: 6px; border-radius: 3px;
  background: rgba(0,0,0,0.08); margin: 10px 0 0; overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(.16,1,.3,1); width: 0; }
.danger  .conf-fill { background: linear-gradient(90deg, var(--red), #e57373); }
.warning .conf-fill { background: linear-gradient(90deg, var(--orange), #ffb74d); }
.safe    .conf-fill { background: linear-gradient(90deg, var(--green-main), var(--green-light)); }

.result-body { background: white; padding: 18px; }

.kannada-box {
  background: #e8f5e8; border: 1px solid var(--green-pale);
  border-radius: 12px; padding: 14px 16px; margin-bottom: 16px;
}
.kannada-label { font-size: 10px; font-weight: 700; color: var(--green-mid); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.kannada-text {
  font-family: 'Noto Sans Kannada', sans-serif;
  font-size: 15px; font-weight: 600; color: #1b5e20; line-height: 1.6;
}

.section-title {
  font-size: 11px; font-weight: 700; color: var(--text-light);
  text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 10px; margin-top: 16px;
  display: flex; align-items: center; gap: 6px;
}
.section-title:first-child { margin-top: 0; }

.treatment-list { display: flex; flex-direction: column; gap: 8px; }
.treatment-item {
  display: flex; align-items: flex-start; gap: 10px;
  background: var(--surface); border-radius: 10px; padding: 10px 12px;
  border: 1px solid var(--border);
}
.treatment-num {
  background: var(--green-main); color: white;
  width: 22px; height: 22px; border-radius: 50%;
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.treatment-text { font-size: 13px; color: var(--text-dark); line-height: 1.4; }

.other-preds { display: flex; flex-direction: column; gap: 6px; }
.pred-row { display: flex; align-items: center; gap: 8px; }
.pred-name { font-size: 12px; color: var(--text-mid); width: 140px; flex-shrink: 0; }
.pred-bar-track { flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.pred-bar-fill  { height: 100%; background: var(--green-light); border-radius: 3px; }
.pred-pct { font-size: 11px; color: var(--text-light); font-family: monospace; width: 36px; text-align: right; }

.kvk-box {
  background: var(--green-dark); border-radius: 14px;
  padding: 14px 16px; margin-top: 16px;
  display: flex; align-items: center; gap: 12px;
}
.kvk-icon { font-size: 28px; }
.kvk-title { font-size: 13px; font-weight: 700; color: #7ed87e; }
.kvk-number { font-size: 18px; font-weight: 800; color: white; letter-spacing: 1px; }
.kvk-hint { font-size: 10px; color: rgba(255,255,255,0.5); font-family: 'Noto Sans Kannada'; }

.share-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.btn-share {
  padding: 12px; border: none; border-radius: 12px;
  font-family: 'Baloo 2', sans-serif; font-size: 13px; font-weight: 700;
  cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all 0.18s; -webkit-tap-highlight-color: transparent;
}
.btn-share:active { transform: scale(0.97); }
.btn-whatsapp { background: #25d366; color: white; }
.btn-rescan   { background: var(--surface); color: var(--green-mid); border: 2px solid var(--border); }

/* ── HISTORY ── */
.history-card { background: white; margin: 0 16px 16px; border-radius: 20px; overflow: hidden; border: 1px solid var(--border); }
.history-header { background: var(--surface); padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 700; color: var(--text-mid); display: flex; align-items: center; gap: 6px; }
.history-item { padding: 12px 16px; border-bottom: 1px solid #f0f7f0; display: flex; align-items: center; gap: 12px; }
.history-item:last-child { border-bottom: none; }
.history-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.history-dot.danger  { background: var(--red); }
.history-dot.warning { background: var(--orange); }
.history-dot.safe    { background: var(--green-main); }
.history-disease { font-size: 13px; font-weight: 700; color: var(--text-dark); }
.history-meta    { font-size: 11px; color: var(--text-light); }
.history-conf    { margin-left: auto; font-size: 12px; font-weight: 700; font-family: monospace; color: var(--green-mid); }

/* ── DEMO BADGE ── */
.demo-notice {
  margin: 0 16px 16px;
  background: #fff8e1; border: 1px solid #ffe082;
  border-radius: 12px; padding: 10px 14px;
  font-size: 12px; color: #5d4037;
  display: flex; align-items: center; gap: 8px;
}

/* ── NO-IMAGE STATE ── */
.no-image-notice {
  display: none;
  background: #ffebee; border: 1px solid #ffcdd2;
  border-radius: 10px; padding: 10px 14px;
  font-size: 13px; color: var(--red);
  margin-top: 10px;
  text-align: center;
}
</style>
</head>

<body>
<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-logo">🌾</div>
  <div>
    <div class="topbar-name">KrishiAI</div>
    <div class="topbar-sub">ಬೆಳೆ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ</div>
  </div>
  <div class="topbar-right">
    <div class="mode-pill" id="mode-pill">LOADING...</div>
  </div>
</div>

<div class="scroll-wrap">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-title">📸 ಫೋಟೋ ತೆಗೆದು ರೋಗ ತಿಳಿಯಿರಿ</div>
    <div class="hero-kannada">Take a photo of your crop leaf to detect disease</div>
    <div class="hero-steps">
      <div class="hero-step">📷 Photo</div>
      <div class="hero-step">🤖 AI Scan</div>
      <div class="hero-step">💊 Treatment</div>
    </div>
  </div>

  <!-- DEMO NOTICE (shown when running without real model) -->
  <div class="demo-notice" id="demo-notice" style="display:none">
    ⚠️ Running in <strong>demo mode</strong> — train the AI model first (step2_train.py), then restart the server.
  </div>

  <!-- MAIN UPLOAD CARD -->
  <div class="card">
    <div class="card-header">
      <span class="card-header-icon">🔬</span>
      <div>
        <div class="card-header-title">Scan Your Crop</div>
        <div class="card-header-sub">ನಿಮ್ಮ ಬೆಳೆಯ ಎಲೆ ಫೋಟೋ ತೆಗೆಯಿರಿ</div>
      </div>
    </div>
    <div class="card-body">

      <!-- Crop type selector -->
      <div style="margin-bottom:14px">
        <div class="input-label">Crop Type <span class="label-kn">ಬೆಳೆ ಆಯ್ಕೆ</span></div>
        <div class="crop-grid">
          <button class="crop-btn active" onclick="selectCrop(this,'paddy')">
            <span class="crop-btn-icon">🌾</span>
            <span class="crop-btn-name">Paddy</span>
            <span class="crop-btn-kannada">ಭತ್ತ</span>
          </button>
          <button class="crop-btn" onclick="selectCrop(this,'coconut')">
            <span class="crop-btn-icon">🥥</span>
            <span class="crop-btn-name">Coconut</span>
            <span class="crop-btn-kannada">ತೆಂಗು</span>
          </button>
          <button class="crop-btn" onclick="selectCrop(this,'areca')">
            <span class="crop-btn-icon">🌴</span>
            <span class="crop-btn-name">Areca</span>
            <span class="crop-btn-kannada">ಅಡಿಕೆ</span>
          </button>
        </div>
      </div>

      <!-- Upload zone -->
      <div class="upload-zone" id="upload-zone">
        <input type="file" accept="image/*" id="file-input" onchange="handleFile(event)">
        <img class="preview-img" id="preview-img">
        <div class="preview-overlay" id="preview-overlay">📸 Tap to change</div>
        <span class="upload-icon" id="up-icon">🌿</span>
        <div class="upload-text" id="up-text">Tap to select a photo</div>
        <div class="upload-hint" id="up-hint">ಫೋಟೋ ಆಯ್ಕೆ ಮಾಡಿ ಅಥವಾ ತೆಗೆಯಿರಿ</div>
      </div>

      <!-- Camera / Gallery buttons -->
      <div class="capture-row">
        <button class="btn-camera primary" onclick="openCamera()">
          <span class="cam-icon">📷</span> Camera
        </button>
        <button class="btn-camera secondary" onclick="openGallery()">
          <span class="cam-icon">🖼️</span> Gallery
        </button>
      </div>

      <!-- Hidden camera input -->
      <input type="file" accept="image/*" capture="environment" id="camera-input" style="display:none" onchange="handleFile(event)">
      <input type="file" accept="image/*"                        id="gallery-input" style="display:none" onchange="handleFile(event)">

      <!-- No-image warning -->
      <div class="no-image-notice" id="no-image-notice">
        ⚠️ Please take or select a photo first! <br>
        <span style="font-family:'Noto Sans Kannada'">ಮೊದಲು ಫೋಟೋ ತೆಗೆಯಿರಿ</span>
      </div>

      <!-- Farmer info -->
      <div style="margin-top:18px; border-top:1px solid var(--border); padding-top:16px">
        <div class="input-row">
          <label class="input-label" for="farmer-name">
            Your Name (optional) <span class="label-kn">ನಿಮ್ಮ ಹೆಸರು</span>
          </label>
          <input class="input-field" type="text" id="farmer-name"
                 placeholder="ರಾಮಣ್ಣ ಗೌಡ / Ramanna Gowda"
                 autocomplete="name">
        </div>
        <div class="input-row">
          <label class="input-label" for="plot-name">
            Plot / Village <span class="label-kn">ಜಮೀನು / ಊರು</span>
          </label>
          <input class="input-field" type="text" id="plot-name"
                 placeholder="e.g. Manipal Road Plot A1"
                 autocomplete="off">
        </div>
      </div>

      <!-- Analyze button -->
      <button class="btn-analyze" id="analyze-btn" onclick="runAnalysis()">
        🔍 Analyze Crop &nbsp;<span class="kn">ರೋಗ ಪತ್ತೆ ಮಾಡಿ</span>
      </button>

    </div>
  </div>

  <!-- LOADING STATE -->
  <div class="loading-card" id="loading-card">
    <div class="loading-spinner"></div>
    <div class="loading-text">Analyzing your crop...</div>
    <div class="loading-sub">ನಿಮ್ಮ ಬೆಳೆ ತಪಾಸಣೆ ನಡೆಯುತ್ತಿದೆ...</div>
  </div>

  <!-- RESULT CARD -->
  <div class="result-card" id="result-card">
    <!-- header injected by JS -->
  </div>

  <!-- HISTORY -->
  <div class="history-card" id="history-card" style="display:none">
    <div class="history-header">📋 Recent Scans &nbsp;<span style="font-family:'Noto Sans Kannada';font-weight:400;font-size:12px">ಇತ್ತೀಚಿನ ತಪಾಸಣೆ</span></div>
    <div id="history-list"></div>
  </div>

</div><!-- /scroll-wrap -->

<script>
// ── State ──
let selectedCrop    = 'paddy';
let selectedFile    = null;
let scanHistory     = [];

// ── Check server mode on load ──
fetch('/health')
  .then(r => r.json())
  .then(d => {
    const pill = document.getElementById('mode-pill');
    if (d.model_ready) {
      pill.textContent = 'LIVE AI';
      pill.style.background = 'rgba(90,175,90,0.2)';
      pill.style.color = '#7ed87e';
    } else {
      pill.textContent = 'DEMO';
      pill.style.background = 'rgba(244,168,0,0.2)';
      pill.style.color = '#f4a800';
      pill.style.borderColor = 'rgba(244,168,0,0.3)';
      document.getElementById('demo-notice').style.display = 'flex';
    }
  })
  .catch(() => {
    document.getElementById('mode-pill').textContent = 'OFFLINE';
  });

// ── Crop selection ──
function selectCrop(btn, crop) {
  selectedCrop = crop;
  document.querySelectorAll('.crop-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// ── Camera / Gallery ──
function openCamera()  { document.getElementById('camera-input').click(); }
function openGallery() { document.getElementById('gallery-input').click(); }

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  const preview = document.getElementById('preview-img');
  const zone    = document.getElementById('upload-zone');

  preview.src = url;
  preview.style.display = 'block';
  document.getElementById('preview-overlay').style.display = 'block';
  document.getElementById('up-icon').style.display = 'none';
  document.getElementById('up-text').style.display = 'none';
  document.getElementById('up-hint').style.display = 'none';
  zone.classList.add('has-image');
  document.getElementById('no-image-notice').style.display = 'none';
  // Reset any previous result
  document.getElementById('result-card').style.display = 'none';
}

// Also allow tapping the zone to open file picker
document.getElementById('upload-zone').addEventListener('click', function(e) {
  if (e.target.tagName === 'INPUT') return;
  document.getElementById('file-input').click();
});

// ── Main analysis ──
async function runAnalysis() {
  if (!selectedFile) {
    document.getElementById('no-image-notice').style.display = 'block';
    document.getElementById('no-image-notice').scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }

  // Show loading
  document.getElementById('loading-card').style.display = 'block';
  document.getElementById('result-card').style.display  = 'none';
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('loading-card').scrollIntoView({ behavior: 'smooth', block: 'center' });

  try {
    const formData = new FormData();
    formData.append('image',       selectedFile);
    formData.append('farmer_name', document.getElementById('farmer-name').value.trim());
    formData.append('crop_type',   selectedCrop);
    formData.append('plot_name',   document.getElementById('plot-name').value.trim());

    const resp   = await fetch('/predict', { method: 'POST', body: formData });
    const result = await resp.json();

    if (!resp.ok) {
      throw new Error(result.error || 'Server error');
    }

    showResult(result);
    addToHistory(result);

  } catch (err) {
    document.getElementById('loading-card').style.display = 'none';
    alert('Error: ' + err.message + '\n\nMake sure the Flask server is running.');
  } finally {
    document.getElementById('analyze-btn').disabled = false;
  }
}

// ── Render result ──
function showResult(r) {
  document.getElementById('loading-card').style.display = 'none';

  const isHealthy = r.risk_level === 'LOW';
  const isHigh    = r.risk_level === 'HIGH';
  const cls       = isHealthy ? 'safe' : isHigh ? 'danger' : 'warning';
  const icon      = isHealthy ? '✅' : isHigh ? '🚨' : '⚠️';
  const confPct   = parseFloat(r.confidence_pct) || (r.confidence * 100);

  // Treatment items
  const treatments = (r.treatment || []).map((t, i) => `
    <div class="treatment-item">
      <div class="treatment-num">${i+1}</div>
      <div class="treatment-text">${t}</div>
    </div>
  `).join('');

  // Top-k predictions
  const topK = (r.top_k || []).map(p => `
    <div class="pred-row">
      <div class="pred-name">${p.disease}</div>
      <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${(p.confidence*100).toFixed(0)}%"></div></div>
      <div class="pred-pct">${(p.confidence*100).toFixed(0)}%</div>
    </div>
  `).join('');

  // WhatsApp message
  const waMsg = encodeURIComponent(
    `🌾 *KrishiAI Diagnosis*\n\n` +
    `${icon} *${r.disease}*\n` +
    `Confidence: ${r.confidence_pct}\n` +
    `Risk: ${r.risk_level}\n` +
    `Scan ID: ${r.scan_id}\n\n` +
    `📋 *Treatment:*\n` +
    (r.treatment||[]).map(t => `• ${t}`).join('\n') + `\n\n` +
    `🇮🇳 ${r.kannada_alert}\n\n` +
    `_KVK Udupi: 0820-2520842_`
  );

  const card = document.getElementById('result-card');
  card.innerHTML = `
    <div class="result-header ${cls}">
      <div class="result-icon">${icon}</div>
      <div style="flex:1">
        <div class="result-disease">${r.disease}</div>
        <div class="result-confidence">
          Confidence: ${r.confidence_pct} &nbsp;·&nbsp; Risk: ${r.risk_level}
        </div>
        <div class="result-scan-id">Scan #${r.scan_id} · ${r.timestamp}</div>
        <div class="conf-bar ${cls}">
          <div class="conf-fill" id="cf" style="width:0%"></div>
        </div>
      </div>
    </div>
    <div class="result-body">

      <div class="kannada-box">
        <div class="kannada-label">ಕನ್ನಡ ಸಂದೇಶ · Kannada Alert</div>
        <div class="kannada-text">${r.kannada_alert}</div>
      </div>

      <div class="section-title">💊 Treatment Recommendations</div>
      <div class="treatment-list">${treatments}</div>

      ${topK ? `<div class="section-title" style="margin-top:16px">📊 Model Confidence Breakdown</div>
      <div class="other-preds">${topK}</div>` : ''}

      <div class="kvk-box">
        <div class="kvk-icon">📞</div>
        <div>
          <div class="kvk-title">KVK Udupi — Field Officer</div>
          <div class="kvk-number">0820-2520842</div>
          <div class="kvk-hint">ಸಹಾಯಕ್ಕಾಗಿ ಸಂಪರ್ಕಿಸಿ · Call for expert help</div>
        </div>
      </div>

      ${r.demo_mode ? `<div style="margin-top:12px;font-size:11px;color:#9e9e9e;text-align:center">Demo mode — results are simulated</div>` : ''}

      <div class="share-row">
        <a href="https://wa.me/?text=${waMsg}" target="_blank" style="text-decoration:none">
          <button class="btn-share btn-whatsapp" style="width:100%">
            💬 Share on WhatsApp
          </button>
        </a>
        <button class="btn-share btn-rescan" onclick="resetScan()">
          🔄 Scan Again<br><small style="font-family:'Noto Sans Kannada'">ಮತ್ತೆ ತಪಾಸಿಸಿ</small>
        </button>
      </div>

    </div>
  `;

  card.style.display = 'block';
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Animate confidence bar
  setTimeout(() => {
    const cf = document.getElementById('cf');
    if (cf) cf.style.width = confPct.toFixed(0) + '%';
  }, 200);
}

// ── History ──
function addToHistory(r) {
  const isHealthy = r.risk_level === 'LOW';
  const isHigh    = r.risk_level === 'HIGH';
  const dotCls    = isHealthy ? 'safe' : isHigh ? 'danger' : 'warning';

  scanHistory.unshift({
    disease:    r.disease,
    conf:       r.confidence_pct,
    time:       r.timestamp,
    risk:       r.risk_level,
    dot:        dotCls,
  });

  renderHistory();
  document.getElementById('history-card').style.display = 'block';
}

function renderHistory() {
  const list = document.getElementById('history-list');
  list.innerHTML = scanHistory.slice(0, 8).map(h => `
    <div class="history-item">
      <div class="history-dot ${h.dot}"></div>
      <div>
        <div class="history-disease">${h.disease}</div>
        <div class="history-meta">${h.time}</div>
      </div>
      <div class="history-conf">${h.conf}</div>
    </div>
  `).join('');
}

// ── Reset ──
function resetScan() {
  selectedFile = null;
  document.getElementById('preview-img').style.display      = 'none';
  document.getElementById('preview-overlay').style.display  = 'none';
  document.getElementById('up-icon').style.display          = '';
  document.getElementById('up-text').style.display          = '';
  document.getElementById('up-hint').style.display          = '';
  document.getElementById('upload-zone').classList.remove('has-image');
  document.getElementById('result-card').style.display      = 'none';
  document.getElementById('file-input').value               = '';
  document.getElementById('camera-input').value             = '';
  document.getElementById('gallery-input').value            = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    # Find local IP so farmers on the same WiFi can connect
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "localhost"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           KrishiAI — Farmer Web Server                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Model status : {'✅ Live AI' if MODEL_READY else '⚠️  Demo mode (run step2_train.py first)'}")
    print()
    print(f"  Local access   :  http://localhost:5000")
    print(f"  On same WiFi   :  http://{local_ip}:5000")
    print()
    print(f"  For internet access (public URL):")
    print(f"    pip install pyngrok")
    print(f"    ngrok http 5000")
    print()
    print(f"  Farmer opens the URL on their phone → takes a photo → gets diagnosis")
    print()

    app.run(host="0.0.0.0", port=5000, debug=False)
