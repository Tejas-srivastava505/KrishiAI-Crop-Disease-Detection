# KrishiAI — AI Crop Disease Detection System
## Udupi Farmers · EfficientNetB0 + TFLite (Float16) · Kannada Alerts

---

## Project Structure

```
KrishiAI/
├── 02_step1_data_pipeline.ipynb   # Download → Clean → Split → tf.data
├── 03_step2_train.ipynb           # Train EfficientNetB0 (2 phases)
├── 04_step3_evaluate.ipynb        # Confusion matrix, ROC, Grad-CAM
├── 05_step4_export_and_infer.ipynb# TFLite export (Float16) + inference engine
├── requirements.txt
│
├── data/
│   ├── raw/                   # Downloaded PlantVillage images, via kagglehub (by class)
│   └── cleaned/                # After removing corrupt/blurry/duplicate images
│
├── models/
│   ├── best_phase1.keras      # Best model from Phase 1 (head training)
│   ├── best_phase2.keras      # Best model from Phase 2 (fine-tuning)
│   └── krishiai_final.keras   # Final trained model (~33 MB, float32)
│
├── tflite/
│   ├── krishiai_float16.tflite               # ← Deploy this on Android (production)
│   ├── krishiai_efficientnetb0_int8.tflite   # Exported for reference only — NOT used
│   │                                          # (see "Why Float16, not INT8" below)
│   └── model_metadata.json
│
├── logs/
│   ├── class_to_idx.json      # {"Rice___Leaf_blast": 4, ...}
│   ├── idx_to_class.json      # {4: "Rice___Leaf_blast", ...}
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

### Google Colab (recommended)
Each notebook mounts Google Drive and reads/writes directly under
`/content/drive/MyDrive/KrishiAI/...` — no local setup needed. Just open the
notebooks in order and run all cells. A GPU runtime (T4 or better) is
**required** for Step 2 training — Phase 1+2 on CPU can take 1.5–2+ hours
vs. ~35–45 minutes on a T4.

### Running locally instead
```bash
# TensorFlow does not yet support Python 3.14 — use 3.11 or 3.12
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```
Then edit `BASE_DIR` at the top of **each** notebook's config cell (every
notebook pastes its own copy — there is no shared config import) to point at
your local project folder instead of the Colab Drive path, and remove/skip
the `drive.mount(...)` cell.

---

## Run the Full Pipeline

Run the notebooks in order, top to bottom, in a **fresh runtime each time**
you change config values (in-memory state from a previous run can silently
mask edits — see "Lessons Learned" below):

1. `02_step1_data_pipeline.ipynb` — downloads via `kagglehub` (two Kaggle
   sources: one for Corn/Tomato, one for Rice), cleans, and splits into
   stratified train/val/test manifests
2. `03_step2_train.ipynb` — Phase 1 (frozen head) + Phase 2 (fine-tune top 30
   layers)
3. `04_step3_evaluate.ipynb` — confusion matrix, ROC curves, Grad-CAM, full
   classification report
4. `05_step4_export_and_infer.ipynb` — exports the trained model to TFLite
   and runs the inference engine

---

## Key Design Decisions

### Why EfficientNetB0?
- Strong accuracy/efficiency tradeoff, pretrained on ImageNet → converges
  fast with limited data
- **Caveat found during this project:** EfficientNet's Swish activation and
  Squeeze-and-Excitation blocks are known to quantize poorly to full INT8
  (Google's own TF team reported ImageNet accuracy dropping from 75%→46%
  under INT8 PTQ for this reason) — see "Why Float16, not INT8" below

### Data Source
PlantVillage's `tensorflow_datasets` loader was replaced with `kagglehub`
after persistent HTTP 403 errors. Two separate Kaggle datasets are pulled —
one covering Corn/Tomato classes, one covering Rice disease classes — since
several originally-planned rice disease classes don't exist in the standard
PlantVillage mirror.

### The 8 Classes
```python
CLASS_NAMES = [
    "Leaf Blight",            # 0 — Corn___Northern_Leaf_Blight
    "Healthy Crop (Corn)",    # 1 — Corn___healthy
    "Bacterial Blight",       # 2 — Rice___Bacterial_leaf_blight
    "Brown Spot",             # 3 — Rice___Brown_spot
    "Rice Blast",             # 4 — Rice___Leaf_blast
    "Healthy Rice",           # 5 — Rice___healthy
    "Fungal Blight",          # 6 — Tomato___Late_blight
    "Healthy Crop (Tomato)",  # 7 — Tomato___healthy
]
```
This list — and its order — must match `class_to_idx.json` exactly (index 0
through 7). It's defined independently in each of the four notebooks; when
changing it, update **all four**, not just one.

### Data Cleaning Steps
| Check | Threshold | Why |
|-------|-----------|-----|
| File size | < 2 KB → discard | Truncated/corrupt JPEG |
| Pixel size | < 64×64 → discard | Too small for lesion features |
| Blur (Laplacian) | variance < 10 → discard | Phone blur loses disease patterns |
| MD5 duplicate | exact match → discard | Source data has some duplicates |

### Two-Phase Training
| Phase | What trains | LR | Epochs | Purpose |
|-------|-------------|-----|--------|---------|
| Phase 1 | Head only (EfficientNet frozen) | 1e-3 | 10 | Learn crop-disease features fast |
| Phase 2 | Top 30 layers + head | 1e-5 | 20 | Fine-tune for Udupi-specific patterns |

### Augmentation (Monsoon-tuned)
- Brightness: ±30% (overcast → harsh sun)
- Saturation: ±30% (wet leaves vs dry)
- Contrast jitter (leaf texture variation)
- Hue: ±5% (yellowing stages)
- Horizontal flip

Augmentation is applied **after** the decode/resize `.cache()` step (not
before), so every epoch sees freshly randomized augmentation instead of one
frozen augmented version repeated every epoch.

### Why Float16, not INT8
Original target was INT8 quantization for a ≤4 MB model. In practice, full
INT8 post-training quantization caused a severe confidence collapse on this
model (correct predictions, but confidence dropping to ~30%) — a known,
documented failure mode for EfficientNet's Swish activations under INT8.
Float16 was adopted instead:

| Format | Size | Behavior |
|--------|------|----------|
| Keras float32 | 33.1 MB | Baseline, 97% test accuracy |
| **Float16 TFLite (deployed)** | **8.74 MB** | Matches float32 accuracy/confidence closely |
| INT8 TFLite (exported, unused) | 5.25 MB | Confidence collapse — not used for deployment |

Float16 doesn't hit the original 4 MB target, but produces trustworthy
confidence scores, which matters more for a farmer-facing diagnostic tool
than an arbitrary size target. If a future revision needs a smaller
footprint, try **dynamic-range quantization** (int8 weights, float
activations — no representative dataset needed) as a middle ground before
returning to full INT8.

---

## Inference API (importable)

```python
from step4_export_and_infer import KrishiAIInference

engine = KrishiAIInference()   # loads krishiai_float16.tflite by default
result = engine.predict("leaf_photo.jpg")

print(result["disease"])          # "Rice Blast"
print(result["confidence_pct"])   # "93.2%"
print(result["kannada_alert"])    # "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ ಪತ್ತೆ..."
print(result["treatment"])        # ["Tricyclazole 75WP @ 0.6 g/L", ...]
print(result["risk_level"])       # "HIGH"

# WhatsApp-formatted reply
msg = engine.format_whatsapp_reply(result)
```

**Note:** `predict()` uses the model's raw output directly as the
probability distribution (`probs = output`) — the model's final layer
already applies `softmax`, so no additional softmax is applied at inference
time. (An earlier version double-applied softmax here, which silently
compressed genuine ~99% confidence down to ~30% without affecting the
predicted class — worth knowing if this file is ever refactored again.)

---

## Target Metrics vs. Actual Result
| Metric | Original Target | Actual Result |
|--------|------------------|---------------|
| Overall accuracy | ≥ 90% | **97%** |
| Macro F1 | — | ~0.97 |
| Inference speed | < 500 ms | ~40–70 ms (CPU, Float16) |
| Model size | < 4 MB (INT8) | 8.74 MB (Float16 — chosen for accuracy) |
| Early warning window | 5–7 days | via NDVI delta (not yet implemented) |

---

## Lessons Learned (for future maintainers)

1. **Each notebook pastes its own independent config block.** There is no
   shared `config.py` import — a fix made in one notebook's `CLASS_NAMES` /
   `NUM_CLASSES` does **not** propagate to the others. Check all four when
   changing class definitions.
2. **Editing a cell doesn't apply the fix until it's re-run — and until any
   downstream object built from it is recreated.** Several rounds of
   "the fix didn't work" in this project traced back to a stale
   `KrishiAIInference` object or stale in-memory `NUM_CLASSES` from a prior
   run, not the actual code. When in doubt: Restart runtime → Run all.
3. **Writing to Google Drive directly (not to local `/content/...` then
   copying) avoids silent partial-copy failures** — an earlier version of
   this pipeline lost ~43% of the dataset this way with no error thrown.
4. **`tf.keras.applications.efficientnet.preprocess_input` is a no-op** for
   this model family (EfficientNet has its own internal rescaling layer) —
   don't assume it's doing normalization work if you're debugging
   preprocessing mismatches.

---

*KrishiAI · IT-A · 2025-26 · KVK Udupi Partnership*
