# =============================================================================
# config.py — KrishiAI Project Configuration
# All hyperparameters, paths, and class definitions in ONE place.
# Change values here; the rest of the scripts pick them up automatically.
# =============================================================================

import os

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")          # downloaded zips
DATA_CLEAN_DIR  = os.path.join(BASE_DIR, "data", "cleaned")      # after cleaning
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")    # train/val/test splits
MODELS_DIR      = os.path.join(BASE_DIR, "models")               # saved .keras / .h5
TFLITE_DIR      = os.path.join(BASE_DIR, "tflite")               # exported .tflite
LOGS_DIR        = os.path.join(BASE_DIR, "logs")                 # TensorBoard / CSVs
PLOTS_DIR       = os.path.join(BASE_DIR, "plots")                # confusion matrix, curves

for d in [DATA_RAW_DIR, DATA_CLEAN_DIR, DATA_PROC_DIR,
          MODELS_DIR, TFLITE_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── DATASET ───────────────────────────────────────────────────────────────────
# PlantVillage is available via tensorflow_datasets (no Kaggle key needed).
# We filter it down to only classes relevant to Udupi crops.
# The keys below MUST match the class folder names in PlantVillage exactly.

DISEASE_CLASSES = {
    # ── Rice ──────────────────────────────────────────────────────────────────
    "Rice___Leaf_blast":          "Rice Blast",            # Magnaporthe oryzae
    "Rice___Bacterial_leaf_blight": "Bacterial Leaf Blight", # Xanthomonas oryzae
    "Rice___Brown_spot":          "Brown Spot",            # Bipolaris oryzae
    "Rice___healthy":             "Healthy Rice",

    # ── Coconut / proxy (PlantVillage doesn't have coconut; we use the
    #    closest abiotic-stress proxy — corn with nutrient deficiency)
    "Corn_(maize)___Northern_Leaf_Blight": "Leaf Blight (proxy)",
    "Corn_(maize)___healthy":              "Healthy Crop (proxy)",

    # ── Tomato (used as augmentation source for fungal patterns) ──────────────
    "Tomato___Late_blight":       "Fungal Blight (proxy)",
    "Tomato___healthy":           "Healthy Crop (proxy)",
}

# Friendly short labels for plotting (must stay in same order as CLASS_NAMES list)
CLASS_NAMES = [
    "Rice Blast",
    "Bacterial Blight",
    "Brown Spot",
    "Healthy Rice",
    "Leaf Blight",
    "Fungal Blight",
    "Healthy Crop",
]

NUM_CLASSES = len(CLASS_NAMES)

# ── MODEL ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)   # EfficientNetB0 native input size
BATCH_SIZE   = 32
EPOCHS_FROZEN  = 10        # Phase 1: train only the classification head
EPOCHS_FINETUNE = 20       # Phase 2: unfreeze top layers and fine-tune
LEARNING_RATE_HEAD    = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE = 0.3
L2_LAMBDA    = 1e-4

# ── DATA SPLIT ────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# ── DATA CLEANING THRESHOLDS ──────────────────────────────────────────────────
MIN_IMAGE_SIZE_BYTES = 2_000        # discard images smaller than 2 KB (corrupt)
MIN_PIXEL_DIMENSION  = 64           # discard images narrower/shorter than 64px
MAX_BLUR_VARIANCE    = 10.0         # discard images with Laplacian variance < 10
                                    # (too blurry to be useful)

# ── AUGMENTATION ──────────────────────────────────────────────────────────────
# Tuned for monsoon / field conditions in Karnataka
AUGMENT_CONFIG = {
    "rotation_range":     20,        # ±20° for handheld phone shots
    "zoom_range":         0.2,       # simulate distance variation
    "horizontal_flip":    True,
    "brightness_range":   (0.7, 1.3),# monsoon overcast → harsh afternoon sun
    "width_shift_range":  0.1,
    "height_shift_range": 0.1,
    "shear_range":        0.1,
    "fill_mode":          "reflect",
}

# ── TFLITE EXPORT ─────────────────────────────────────────────────────────────
TFLITE_MODEL_NAME  = "krishiai_efficientnetb0_int8.tflite"
TFLITE_TARGET_MB   = 4.0   # must stay under 4 MB for ₹8k Android phones

# ── KANNADA MESSAGES ──────────────────────────────────────────────────────────
# Shown in the inference output alongside the English diagnosis
KANNADA_ALERTS = {
    "Rice Blast":          "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ ಪತ್ತೆ. ಟ್ರೈಸೈಕ್ಲಾಜ಼ೋಲ್ ತಕ್ಷಣ ಸಿಂಪಡಿಸಿ.",
    "Bacterial Blight":    "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಒಣಗುವಿಕೆ. ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಬಳಸಿ.",
    "Brown Spot":          "ಕಂದು ಚುಕ್ಕೆ ರೋಗ. ಮ್ಯಾಂಕೋಜ಼ೆಬ್ ಬಳಸಿ.",
    "Healthy Rice":        "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
    "Leaf Blight":         "ಎಲೆ ಒಣಗುವ ರೋಗ. KVK ಸಂಪರ್ಕಿಸಿ.",
    "Fungal Blight":       "ಶಿಲೀಂಧ್ರ ರೋಗ ಪತ್ತೆ. ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ.",
    "Healthy Crop":        "ಬೆಳೆ ಆರೋಗ್ಯಕರ. ನಿಯಮಿತ ತಪಾಸಣೆ ಮಾಡಿ.",
}

TREATMENT_MAP = {
    "Rice Blast":       ["Tricyclazole 75WP @ 0.6 g/L", "Isoprothiolane 40EC @ 1.5 mL/L", "Avoid excess nitrogen", "Drain stagnant water"],
    "Bacterial Blight": ["Copper Oxychloride 50WP @ 3 g/L", "Streptocycline @ 0.5 g/L", "Avoid flood irrigation", "Remove infected debris"],
    "Brown Spot":       ["Mancozeb 75WP @ 2.5 g/L", "Propiconazole 25EC @ 1 mL/L", "Correct potassium deficiency"],
    "Healthy Rice":     ["Continue regular monitoring", "Preventive fungicide (optional at tillering)"],
    "Leaf Blight":      ["Copper-based fungicide", "Consult KVK Udupi: 0820-2520842"],
    "Fungal Blight":    ["Mancozeb + Carbendazim combo", "Improve field drainage"],
    "Healthy Crop":     ["Continue monitoring", "Check weekly during monsoon"],
}
