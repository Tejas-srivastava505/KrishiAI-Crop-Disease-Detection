# =============================================================================
# step2_train.py — Build EfficientNetB0 + Train in Two Phases
#
# Architecture:
#   EfficientNetB0 (pretrained ImageNet) → GlobalAveragePooling → Dropout
#   → Dense(256, relu) → Dropout → Dense(NUM_CLASSES, softmax)
#
# Training strategy (standard transfer learning):
#   Phase 1  — Freeze EfficientNetB0 base; train only the new head (10 epochs)
#   Phase 2  — Unfreeze top ~30 layers of base; fine-tune at very low LR (20 epochs)
#
# Run: python step2_train.py
# =============================================================================

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")


# ──────────────────────────────────────────────────────────────────────────────
# REBUILD DATASETS FROM SAVED MANIFESTS
# (So step2 can run independently from step1 once data is prepared.)
# ──────────────────────────────────────────────────────────────────────────────

def load_datasets_from_manifests():
    """Load the train/val/test manifests saved by step1 and rebuild tf.data."""
    AUTOTUNE = tf.data.AUTOTUNE

    def pairs_from_csv(name):
        df = pd.read_csv(os.path.join(LOGS_DIR, f"{name}_manifest.csv"))
        return list(zip(df["path"].tolist(), df["label"].tolist()))

    def make_ds(pairs, augment=False, shuffle=False):
        paths  = [p for p, _ in pairs]
        labels = [l for _, l in pairs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(len(pairs), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: _load_image(p, l, augment), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds, len(pairs)

    train_ds, n_train = make_ds(pairs_from_csv("train"), augment=True,  shuffle=True)
    val_ds,   n_val   = make_ds(pairs_from_csv("val"),   augment=False, shuffle=False)
    test_ds,  n_test  = make_ds(pairs_from_csv("test"),  augment=False, shuffle=False)

    print(f"  train={n_train}  val={n_val}  test={n_test}")
    return train_ds, val_ds, test_ds


@tf.function
def _load_image(path, label, augment):
    """Load, decode, preprocess (EfficientNet expects [0,255] float32)."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)   # EfficientNet preprocessor handles normalisation

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)                # rotation proxy
        img = tf.image.random_brightness(img, max_delta=0.25 * 255)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.clip_by_value(img, 0.0, 255.0)

    # EfficientNet built-in preprocessor (scales to [-1, 1] internally)
    img = preprocess_input(img)
    label_one_hot = tf.one_hot(label, NUM_CLASSES)
    return img, label_one_hot


# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, trainable_base: bool = False):
    """
    Build the KrishiAI classification model.

    Architecture:
        Input (224x224x3)
        → EfficientNetB0 base (pretrained, optionally frozen)
        → GlobalAveragePooling2D          # reduces spatial dims
        → BatchNormalization
        → Dropout(0.3)
        → Dense(256, relu, L2)            # custom head
        → BatchNormalization
        → Dropout(0.3)
        → Dense(NUM_CLASSES, softmax)     # disease probability

    Args:
        trainable_base: If False (Phase 1), all EfficientNetB0 weights are frozen.
                        If True  (Phase 2), top layers are unfrozen for fine-tuning.
    """
    # ── Base model ────────────────────────────────────────────────────────────
    base = EfficientNetB0(
        weights="imagenet",          # use ImageNet pretrained weights
        include_top=False,           # remove ImageNet classification head
        input_shape=(*IMAGE_SIZE, 3)
    )

    if not trainable_base:
        # Phase 1: Freeze the entire base
        base.trainable = False
        print("  Base frozen (Phase 1 — head-only training)")
    else:
        # Phase 2: Unfreeze only the top ~30 layers
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        trainable_count = sum(1 for l in base.layers if l.trainable)
        print(f"  Base partially unfrozen ({trainable_count}/{len(base.layers)} layers trainable, Phase 2)")

    # ── Build full model with Functional API ─────────────────────────────────
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="crop_image")

    # Feature extraction
    x = base(inputs, training=False)          # training=False keeps BN frozen in Phase 1
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)

    # Classification head
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="dense_head"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="KrishiAI_EfficientNetB0")
    return model, base


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(phase: int) -> list:
    """
    Returns the list of Keras callbacks for a given training phase.
    phase=1 → head training | phase=2 → fine-tuning
    """
    checkpoint_path = os.path.join(MODELS_DIR, f"best_phase{phase}.keras")

    callbacks = [
        # Save the best model by validation accuracy
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # Reduce learning rate if val_loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Stop early if val_loss doesn't improve for 7 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Log metrics to CSV
        CSVLogger(
            os.path.join(LOGS_DIR, f"phase{phase}_training_log.csv"),
            append=False
        ),
        # TensorBoard logs
        TensorBoard(
            log_dir=os.path.join(LOGS_DIR, f"tensorboard_phase{phase}"),
            histogram_freq=1
        ),
    ]
    return callbacks, checkpoint_path


# ──────────────────────────────────────────────────────────────────────────────
# LOAD CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def load_class_weights():
    path = os.path.join(LOGS_DIR, "class_weights.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    else:
        print("  [WARN] class_weights.json not found — using uniform weights")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ──────────────────────────────────────────────────────────────────────────────

def plot_history(history, phase_label: str, filename: str):
    """Save loss + accuracy curves for one training phase."""
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a150a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1f0f")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color("#1a3d1a")

    # Loss
    ax1.plot(epochs, history.history["loss"],     color="#7ec850", lw=2, label="Train Loss")
    ax1.plot(epochs, history.history["val_loss"], color="#ff8c00", lw=2, linestyle="--", label="Val Loss")
    ax1.set_title(f"Loss — {phase_label}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(labelcolor="white", framealpha=0.2)
    ax1.grid(alpha=0.2, color="#1a3d1a")

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"],     color="#7ec850", lw=2, label="Train Acc")
    ax2.plot(epochs, history.history["val_accuracy"], color="#ff8c00", lw=2, linestyle="--", label="Val Acc")
    ax2.set_title(f"Accuracy — {phase_label}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(labelcolor="white", framealpha=0.2)
    ax2.grid(alpha=0.2, color="#1a3d1a")

    plt.suptitle(f"KrishiAI Training — {phase_label}", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        KrishiAI — Model Training (Step 2 of 4)          ║")
    print("║     EfficientNetB0 Transfer Learning — Two Phases       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/2] Loading datasets from manifests...")
    train_ds, val_ds, test_ds = load_datasets_from_manifests()
    class_weights = load_class_weights()

    # ── PHASE 1: Train the head only ──────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training Classification Head (base frozen)")
    print("="*60)
    print(f"  Epochs: {EPOCHS_FROZEN}  LR: {LEARNING_RATE_HEAD}")

    model, base = build_model(trainable_base=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ]
    )
    model.summary()

    print(f"\n  Trainable params: {model.count_params():,}")
    callbacks_p1, ckpt_p1 = get_callbacks(phase=1)
    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1,
    )
    t1 = time.time()
    print(f"\n  Phase 1 complete in {(t1-t0)/60:.1f} min")
    print(f"  Best val accuracy: {max(hist1.history['val_accuracy']):.4f}")
    plot_history(hist1, "Phase 1 — Head Only", "phase1_training_curves.png")

    # Load best Phase 1 checkpoint
    print(f"\n  Loading best Phase 1 checkpoint: {ckpt_p1}")
    model = keras.models.load_model(ckpt_p1)

    # ── PHASE 2: Fine-tune top layers ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Fine-tuning Top Layers (partial unfreeze)")
    print("="*60)
    print(f"  Epochs: {EPOCHS_FINETUNE}  LR: {LEARNING_RATE_FINETUNE}")

    # Unfreeze top 30 layers of the EfficientNetB0 base
    base_layer = model.get_layer("efficientnetb0")
    base_layer.trainable = True
    for layer in base_layer.layers[:-30]:
        layer.trainable = False

    # Re-compile with much smaller LR to avoid catastrophic forgetting
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
            keras.metrics.AUC(name="auc", multi_label=False),
        ]
    )

    callbacks_p2, ckpt_p2 = get_callbacks(phase=2)
    t2 = time.time()
    hist2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=callbacks_p2,
        class_weight=class_weights,
        verbose=1,
    )
    t3 = time.time()
    print(f"\n  Phase 2 complete in {(t3-t2)/60:.1f} min")
    print(f"  Best val accuracy: {max(hist2.history['val_accuracy']):.4f}")
    plot_history(hist2, "Phase 2 — Fine-tuning", "phase2_training_curves.png")

    # ── Save final model ──────────────────────────────────────────────────────
    final_model = keras.models.load_model(ckpt_p2)
    final_path  = os.path.join(MODELS_DIR, "krishiai_final.keras")
    final_model.save(final_path)
    print(f"\n  Final model saved → {final_path}")

    # ── Quick evaluation on val set ───────────────────────────────────────────
    print("\n  Quick evaluation on validation set:")
    val_results = final_model.evaluate(val_ds, verbose=0)
    for name, val in zip(final_model.metrics_names, val_results):
        print(f"    {name:<25} {val:.4f}")

    # Save combined history
    all_history = {
        "phase1_accuracy": hist1.history["accuracy"],
        "phase1_val_accuracy": hist1.history["val_accuracy"],
        "phase1_loss": hist1.history["loss"],
        "phase1_val_loss": hist1.history["val_loss"],
        "phase2_accuracy": hist2.history["accuracy"],
        "phase2_val_accuracy": hist2.history["val_accuracy"],
        "phase2_loss": hist2.history["loss"],
        "phase2_val_loss": hist2.history["val_loss"],
    }
    with open(os.path.join(LOGS_DIR, "training_history.json"), "w") as f:
        json.dump(all_history, f)

    print(f"\n✅ Training complete!")
    print(f"   Final model → {final_path}")
    print(f"\nNext step: python step3_evaluate.py")
    return final_model


if __name__ == "__main__":
    train()
