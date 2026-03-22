# =============================================================================
# step4_export_and_infer.py — Export to TFLite + Run Inference
#
# Part A: Export
#   • Convert .keras model to TFLite with INT8 post-training quantization
#   • Verify final .tflite size is under TARGET_MB (4 MB)
#   • Benchmark inference speed on CPU (simulating ₹8k Android phone)
#
# Part B: Inference (can be used standalone)
#   • Load the .tflite model
#   • Accept a crop image path
#   • Return disease name, confidence, Kannada alert, treatment
#
# Run: python step4_export_and_infer.py
# Run inference only: python step4_export_and_infer.py --infer path/to/leaf.jpg
# =============================================================================

import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import *


# ──────────────────────────────────────────────────────────────────────────────
# PART A — EXPORT TO TFLite WITH INT8 QUANTIZATION
# ──────────────────────────────────────────────────────────────────────────────

def representative_data_generator():
    """
    Required by INT8 quantization: ~100–200 representative samples from
    the training distribution so the quantizer can calibrate scale factors.
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(LOGS_DIR, "train_manifest.csv"))
    sample_paths = df["path"].sample(min(200, len(df)), random_state=42).tolist()

    for path in sample_paths:
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE).astype(np.float32)
        img = preprocess_input(img)
        yield [img[np.newaxis, ...]]      # shape: (1, 224, 224, 3)


def export_tflite():
    """
    Convert the final Keras model to TFLite with INT8 quantization.

    Why INT8?
        - Float32 model:  ~20-25 MB  — too large for offline phone storage
        - INT8 quantized: ~4-6 MB   — fits ₹8k Android phones
        - Speed:          ~4x faster inference vs float32 on mobile CPU

    The quantization process:
        1. Converter takes the trained Keras model
        2. representative_data_generator() provides calibration data
        3. Activations and weights are mapped from float32 → int8
        4. Small accuracy drop expected (~0.5–1%) in exchange for speed
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   KrishiAI — TFLite Export + Quantization (Step 4 of 4) ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    model_path = os.path.join(MODELS_DIR, "krishiai_final.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun step2_train.py first.")

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    # ── Step 1: Float16 quantization (intermediate, no calibration data needed) ──
    print("\n[1/3] Exporting Float16 TFLite model...")
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()

    f16_path = os.path.join(TFLITE_DIR, "krishiai_float16.tflite")
    with open(f16_path, "wb") as f:
        f.write(tflite_f16)
    f16_size = os.path.getsize(f16_path) / 1e6
    print(f"  Float16 model: {f16_size:.2f} MB → {f16_path}")

    # ── Step 2: INT8 full quantization (smaller + faster, needs calibration) ──
    print("\n[2/3] Exporting INT8 quantized TFLite model (this takes ~2-5 min)...")
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_generator
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type  = tf.uint8    # input  expects uint8
    converter_int8.inference_output_type = tf.uint8    # output returns uint8

    tflite_int8 = converter_int8.convert()
    int8_path = os.path.join(TFLITE_DIR, TFLITE_MODEL_NAME)
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    int8_size = os.path.getsize(int8_path) / 1e6
    print(f"  INT8 model   : {int8_size:.2f} MB → {int8_path}")

    # ── Step 3: Verify size target ────────────────────────────────────────────
    print(f"\n[3/3] Size check:")
    print(f"  Original Keras model : {os.path.getsize(model_path)/1e6:.1f} MB")
    print(f"  Float16 TFLite       : {f16_size:.2f} MB")
    print(f"  INT8 TFLite          : {int8_size:.2f} MB  (target ≤ {TFLITE_TARGET_MB} MB)")
    if int8_size <= TFLITE_TARGET_MB:
        print(f"  ✅ Size target MET — suitable for ₹8,000 Android phones")
    else:
        print(f"  ⚠️  Size {int8_size:.2f} MB exceeds {TFLITE_TARGET_MB} MB target")
        print(f"     Consider: reduce IMAGE_SIZE, use EfficientNetB0 lite variant")

    # ── Benchmark inference speed on CPU ──────────────────────────────────────
    print("\nBenchmarking inference speed (INT8 model, CPU)...")
    _benchmark_tflite(int8_path, n_runs=50, use_uint8=True)

    # ── Save metadata alongside the model ─────────────────────────────────────
    with open(os.path.join(LOGS_DIR, "idx_to_class.json")) as f:
        idx_to_class = json.load(f)

    metadata = {
        "model_file":    TFLITE_MODEL_NAME,
        "image_size":    list(IMAGE_SIZE),
        "num_classes":   NUM_CLASSES,
        "class_labels":  idx_to_class,
        "input_type":    "uint8",
        "input_range":   [0, 255],
        "model_size_mb": round(int8_size, 3),
        "quantization":  "INT8",
        "language":      "kn,en",
    }
    meta_path = os.path.join(TFLITE_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nModel metadata saved → {meta_path}")

    print(f"\n✅ Export complete!")
    print(f"   INT8 TFLite → {int8_path}")
    return int8_path


def _benchmark_tflite(tflite_path: str, n_runs: int = 50, use_uint8: bool = False):
    """
    Measure average inference time per image.
    Simulates the ₹8k Android phone scenario (single-threaded CPU).
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create a dummy image
    dtype = np.uint8 if use_uint8 else np.float32
    dummy = np.random.randint(0, 255, (1, *IMAGE_SIZE, 3)).astype(dtype)

    # Warmup
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    print(f"  Inference time: {mean_ms:.1f} ± {std_ms:.1f} ms  (over {n_runs} runs)")
    print(f"  Target: <500 ms — {'✅ MET' if mean_ms < 500 else '⚠️ Exceeds target'}")


# ──────────────────────────────────────────────────────────────────────────────
# PART B — INFERENCE ENGINE
# Can be imported by the Flask/FastAPI WhatsApp bot or run standalone.
# ──────────────────────────────────────────────────────────────────────────────

class KrishiAIInference:
    """
    Lightweight inference wrapper around the INT8 TFLite model.
    Designed to be imported by the WhatsApp bot backend.

    Usage:
        engine = KrishiAIInference()
        result = engine.predict("path/to/leaf_image.jpg")
        print(result)
    """

    def __init__(self, tflite_path: str = None):
        if tflite_path is None:
            tflite_path = os.path.join(TFLITE_DIR, TFLITE_MODEL_NAME)
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(
                f"TFLite model not found: {tflite_path}\n"
                "Run step4_export_and_infer.py first."
            )

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Determine input dtype (float32 for F16 model, uint8 for INT8)
        self.input_dtype  = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']

        # Load class labels
        meta_path = os.path.join(TFLITE_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.idx_to_class = {int(k): v for k, v in meta["class_labels"].items()}
        else:
            self.idx_to_class = {i: CLASS_NAMES[i] for i in range(NUM_CLASSES)}

        print(f"✅ KrishiAI Inference Engine loaded")
        print(f"   Model: {os.path.basename(tflite_path)}")
        print(f"   Classes: {len(self.idx_to_class)}")

    def preprocess(self, image_input) -> np.ndarray:
        """
        Accepts:
            • File path (str or Path)
            • NumPy array (H, W, 3) BGR or RGB
            • PIL Image

        Returns: preprocessed array ready for the TFLite interpreter.
        """
        if isinstance(image_input, (str, os.PathLike)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Cannot read image: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            # Assume BGR from cv2
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[-1] == 3 else image_input
        else:
            # PIL image
            img_rgb = np.array(image_input.convert("RGB"))

        # Resize to model input size
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE).astype(np.float32)  # (224, 224, 3)

        if self.input_dtype == np.uint8:
            # INT8 quantized model expects [0, 255] uint8
            processed = img_resized.astype(np.uint8)
        else:
            # Float model: apply EfficientNet normalization
            processed = preprocess_input(img_resized)

        return processed[np.newaxis, ...]   # (1, 224, 224, 3)

    def predict(self, image_input, top_k: int = 3) -> dict:
        """
        Run inference on one image.

        Returns dict:
            {
              "disease":     "Rice Blast",
              "confidence":  0.93,
              "top_k":       [{"disease": "Rice Blast", "confidence": 0.93}, ...],
              "kannada_alert": "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ ಪತ್ತೆ...",
              "treatment":   ["Tricyclazole 75WP @ 0.6 g/L", ...],
              "risk_level":  "HIGH",
              "inference_ms": 312.4,
            }
        """
        # Preprocess
        input_tensor = self.preprocess(image_input)

        # Inference
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        # De-quantize if INT8 output
        if self.output_dtype == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)

        # Softmax probabilities
        probs = np.exp(output) / np.sum(np.exp(output)) if output.max() < 1.5 else output

        # Top-k classes
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            {
                "disease":    self.idx_to_class.get(i, f"Class_{i}"),
                "confidence": round(float(probs[i]), 4)
            }
            for i in top_k_indices
        ]

        top_class = top_k_results[0]["disease"]
        top_conf  = top_k_results[0]["confidence"]

        # Risk level
        if top_conf >= 0.85 and "healthy" not in top_class.lower():
            risk = "HIGH"
        elif top_conf >= 0.65 and "healthy" not in top_class.lower():
            risk = "MODERATE"
        elif "healthy" in top_class.lower():
            risk = "LOW"
        else:
            risk = "UNCERTAIN — re-scan or consult KVK"

        # Kannada alert and treatment recommendations
        kannada = KANNADA_ALERTS.get(top_class, "ಕೃಷಿ ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ.")
        treatment = TREATMENT_MAP.get(top_class, ["Consult KVK Udupi: 0820-2520842"])

        return {
            "disease":       top_class,
            "confidence":    top_conf,
            "confidence_pct": f"{top_conf*100:.1f}%",
            "top_k":         top_k_results,
            "kannada_alert": kannada,
            "treatment":     treatment,
            "risk_level":    risk,
            "inference_ms":  round(inference_ms, 1),
        }

    def predict_batch(self, image_paths: list) -> list:
        """Run predict() on a list of images. Returns list of result dicts."""
        return [self.predict(p) for p in image_paths]

    def format_whatsapp_reply(self, result: dict) -> str:
        """Format a prediction result as a WhatsApp-ready text message."""
        risk_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(result["risk_level"], "⚪")
        lines = [
            f"🌾 *KrishiAI Diagnosis*",
            f"",
            f"{risk_emoji} *{result['disease']}*",
            f"Confidence: {result['confidence_pct']}",
            f"Risk: {result['risk_level']}",
            f"",
            f"📋 *Treatment:*",
        ]
        lines += [f"  • {t}" for t in result["treatment"]]
        lines += [
            f"",
            f"🇮🇳 {result['kannada_alert']}",
            f"",
            f"_KVK Udupi: 0820-2520842_"
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# DEMO — print full inference output for one test image
# ──────────────────────────────────────────────────────────────────────────────

def demo_inference(image_path: str):
    """Run inference and print a rich report for a single image."""
    engine = KrishiAIInference()
    print(f"\nAnalyzing: {image_path}")
    print("─" * 50)

    result = engine.predict(image_path)

    print(f"Top prediction : {result['disease']}")
    print(f"Confidence     : {result['confidence_pct']}")
    print(f"Risk level     : {result['risk_level']}")
    print(f"Inference time : {result['inference_ms']} ms")
    print(f"\nTop-{len(result['top_k'])} predictions:")
    for r in result['top_k']:
        bar = "█" * int(r['confidence'] * 20)
        print(f"  {r['disease']:<30} {r['confidence']*100:>5.1f}%  {bar}")

    print(f"\nTreatment recommendations:")
    for t in result['treatment']:
        print(f"  → {t}")

    print(f"\nKannada alert:")
    print(f"  {result['kannada_alert']}")

    print(f"\nWhatsApp message preview:")
    print("─" * 50)
    print(engine.format_whatsapp_reply(result))
    print("─" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KrishiAI TFLite Export & Inference")
    parser.add_argument("--infer", type=str, default=None,
                        help="Path to a leaf/crop image to run inference on")
    args = parser.parse_args()

    if args.infer:
        # Inference only — assumes TFLite model already exported
        demo_inference(args.infer)
    else:
        # Export TFLite model, then run a demo inference on a test image
        tflite_path = export_tflite()

        # Run a demo on the first test image we can find
        import pandas as pd
        test_df = pd.read_csv(os.path.join(LOGS_DIR, "test_manifest.csv"))
        if len(test_df) > 0:
            demo_path = test_df.iloc[0]["path"]
            print(f"\n{'='*60}")
            print("DEMO INFERENCE on first test image:")
            demo_inference(demo_path)

        print(f"\n✅ All done!")
        print(f"   TFLite model : {tflite_path}")
        print(f"   To infer     : python step4_export_and_infer.py --infer your_image.jpg")
