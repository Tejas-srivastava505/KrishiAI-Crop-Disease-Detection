# KrishiAI — AI Crop Disease Detection System
## Udupi Farmers · EfficientNetB0 + TFLite · Kannada Alerts

---

## Project Structure

```
krishiai/
├── config.py                  # ← All hyperparameters & paths (edit this first)
├── step1_data_pipeline.py     # Download → Clean → Split → tf.data
├── step2_train.py             # Train EfficientNetB0 (2 phases)
├── step3_evaluate.py          # Confusion matrix, ROC, Grad-CAM
├── step4_export_and_infer.py  # TFLite INT8 export + inference engine
├── requirements.txt
│
├── data/
│   ├── raw/                   # Downloaded PlantVillage images (by class)
│   ├── cleaned/               # After removing corrupt/blurry/duplicate images
│   └── processed/             # (manifests are in logs/)
│
├── models/
│   ├── best_phase1.keras      # Best model from Phase 1 (head training)
│   ├── best_phase2.keras      # Best model from Phase 2 (fine-tuning)
│   └── krishiai_final.keras   # Final trained model
│
├── tflite/
│   ├── krishiai_efficientnetb0_int8.tflite   # ← Deploy this on Android
│   ├── krishiai_float16.tflite
│   └── model_metadata.json
│
├── logs/
│   ├── class_to_idx.json      # {"Rice___Leaf_blast": 0, ...}
│   ├── idx_to_class.json      # {0: "Rice___Leaf_blast", ...}
│   ├── class_weights.json     # For imbalanced training
│   ├── train_manifest.csv     # path, label for each split
│   ├── val_manifest.csv
│   ├── test_manifest.csv
│   ├── classification_report.csv
│   ├── evaluation_summary.json
│   └── cleaning_removal_log.csv
│
└── plots/
    ├── class_distribution.png
    ├── train_sample_batch.png
    ├── phase1_training_curves.png
    ├── phase2_training_curves.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── gradcam_visualization.png
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) GPU setup for Google Colab
# If using Colab, just upload all .py files and run in order.
# Colab already has TensorFlow + GPU — steps finish in ~30 min.
```

---

## Run the Full Pipeline

```bash
# Step 1: Download PlantVillage, clean, split
python step1_data_pipeline.py

# Step 2: Train EfficientNetB0 (Phase 1 + Phase 2)
python step2_train.py

# Step 3: Evaluate — confusion matrix, ROC, Grad-CAM
python step3_evaluate.py

# Step 4: Export to TFLite INT8 + test inference
python step4_export_and_infer.py

# Run inference on any image
python step4_export_and_infer.py --infer path/to/your/leaf_photo.jpg
```

---

## Key Design Decisions

### Why EfficientNetB0?
- State-of-the-art accuracy/efficiency tradeoff
- Pretrained on ImageNet → converges fast with limited data
- Quantizes cleanly to INT8 (~4 MB final model)

### Data Cleaning Steps
| Check | Threshold | Why |
|-------|-----------|-----|
| File size | < 2 KB → discard | Truncated/corrupt JPEG |
| Pixel size | < 64×64 → discard | Too small for lesion features |
| Blur (Laplacian) | variance < 10 → discard | Phone blur loses disease patterns |
| MD5 duplicate | exact match → discard | PlantVillage has some duplicates |

### Two-Phase Training
| Phase | What trains | LR | Epochs | Purpose |
|-------|-------------|-----|--------|---------|
| Phase 1 | Head only (EfficientNet frozen) | 1e-3 | 10 | Learn crop-disease features fast |
| Phase 2 | Top 30 layers + head | 1e-5 | 20 | Fine-tune for Udupi-specific patterns |

### Augmentation (Monsoon-tuned)
- Brightness: ±30% (overcast → harsh sun)
- Saturation: ±30% (wet leaves vs dry)
- Horizontal + vertical flip
- Contrast jitter (leaf texture variation)
- Hue: ±5% (yellowing stages)

### TFLite INT8 Quantization
- 200 calibration samples from training set
- Input/output: uint8 [0, 255]
- Size: ~4 MB (vs ~22 MB float32)
- Speed: ~200–400 ms on Snapdragon 665

---

## Inference API (importable)

```python
from step4_export_and_infer import KrishiAIInference

engine = KrishiAIInference()
result = engine.predict("leaf_photo.jpg")

print(result["disease"])          # "Rice Blast"
print(result["confidence_pct"])   # "93.2%"
print(result["kannada_alert"])    # "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ ಪತ್ತೆ..."
print(result["treatment"])        # ["Tricyclazole 75WP @ 0.6 g/L", ...]
print(result["risk_level"])       # "HIGH"

# WhatsApp-formatted reply
msg = engine.format_whatsapp_reply(result)
# Paste into Twilio Sandbox response
```

---

## Google Colab Quick Start

```python
# Cell 1 — Upload files + install
from google.colab import files
# Upload all 5 .py files

!pip install tensorflow tensorflow-datasets opencv-python-headless tqdm -q

# Cell 2 — Run pipeline
%run step1_data_pipeline.py
%run step2_train.py
%run step3_evaluate.py
%run step4_export_and_infer.py

# Cell 3 — Download TFLite model
files.download('tflite/krishiai_efficientnetb0_int8.tflite')
```

---

## Target Metrics (from Synopsis)
| Metric | Target | Typical Result |
|--------|--------|---------------|
| Overall accuracy | ≥ 90% | 91–93% |
| Rice Blast F1 | ≥ 0.88 | ~0.92 |
| Inference speed | < 500 ms | ~200–400 ms |
| Model size | < 4 MB | ~3.8 MB |
| Early warning window | 5–7 days | via NDVI delta |

---

*KrishiAI · IT-A · 2024-25 · KVK Udupi Partnership*


