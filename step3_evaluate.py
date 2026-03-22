# =============================================================================
# step3_evaluate.py — Full Evaluation on Held-Out Test Set
#
# Generates:
#   • Per-class precision / recall / F1 (classification report)
#   • Confusion matrix (normalised heatmap)
#   • Grad-CAM visualizations (which part of the leaf the model is looking at)
#   • ROC curves per class
#   • Saves a JSON summary of all metrics
#
# Run: python step3_evaluate.py
# =============================================================================

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, average_precision_score
)

from config import *


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + TEST DATASET
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_test_data():
    final_path = os.path.join(MODELS_DIR, "krishiai_final.keras")
    if not os.path.exists(final_path):
        raise FileNotFoundError(
            f"Model not found at {final_path}.\n"
            "Run step2_train.py first!"
        )

    print(f"Loading model: {final_path}")
    model = keras.models.load_model(final_path)

    # Rebuild test dataset from manifest
    df = pd.read_csv(os.path.join(LOGS_DIR, "test_manifest.csv"))
    paths  = df["path"].tolist()
    labels = df["label"].tolist()

    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: _load_image_eval(p, l), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return model, ds, paths, labels


@tf.function
def _load_image_eval(path, label):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, tf.one_hot(label, NUM_CLASSES)


# ──────────────────────────────────────────────────────────────────────────────
# COLLECT PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_predictions(model, test_ds) -> tuple:
    """
    Run inference on entire test set.
    Returns:
        y_true     : (N,) int32 — ground truth class indices
        y_pred_cls : (N,) int32 — predicted class indices
        y_pred_prob: (N, C) float32 — softmax probabilities
    """
    print("Running inference on test set...")
    all_probs = []
    all_true  = []

    for batch_imgs, batch_labels in test_ds:
        probs = model(batch_imgs, training=False).numpy()
        true  = tf.argmax(batch_labels, axis=1).numpy()
        all_probs.append(probs)
        all_true.append(true)

    y_pred_prob = np.concatenate(all_probs, axis=0)
    y_true      = np.concatenate(all_true,  axis=0)
    y_pred_cls  = np.argmax(y_pred_prob, axis=1)

    return y_true, y_pred_cls, y_pred_prob


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_report(y_true, y_pred_cls):
    """Print sklearn classification report and save to CSV."""
    # Load class index mapping
    with open(os.path.join(LOGS_DIR, "idx_to_class.json")) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    target_names = [idx_to_class.get(i, str(i)) for i in range(NUM_CLASSES)]

    report_text = classification_report(y_true, y_pred_cls, target_names=target_names, digits=4)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report_text)

    # Save as CSV
    report_dict = classification_report(
        y_true, y_pred_cls, target_names=target_names,
        output_dict=True, digits=4
    )
    df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(LOGS_DIR, "classification_report.csv")
    df.to_csv(csv_path)
    print(f"Report saved → {csv_path}")

    # Extract overall accuracy
    accuracy = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    print(f"\n  Overall Accuracy : {accuracy*100:.2f}%")
    print(f"  Macro F1 Score   : {macro_f1:.4f}")

    return accuracy, macro_f1


# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred_cls):
    """Plot normalised confusion matrix."""
    with open(os.path.join(LOGS_DIR, "idx_to_class.json")) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    labels = [idx_to_class.get(i, str(i)) for i in range(NUM_CLASSES)]
    short_labels = [l.replace("_", "\n")[:18] for l in labels]

    cm = confusion_matrix(y_true, y_pred_cls)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0a150a")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (Raw Counts)", "d"),
        (axes[1], cm_norm, "Confusion Matrix (Normalised)", ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=short_labels, yticklabels=short_labels,
            cmap="YlGn", linewidths=0.5, linecolor="#0a150a",
            ax=ax, cbar=True,
            annot_kws={"size": 9}
        )
        ax.set_xlabel("Predicted", color="white", fontsize=11)
        ax.set_ylabel("True",      color="white", fontsize=11)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_facecolor("#0f1f0f")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1a3d1a")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    plt.suptitle("KrishiAI — Test Set Confusion Matrix", color="white", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# ROC CURVES (One-vs-Rest per class)
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_pred_prob):
    with open(os.path.join(LOGS_DIR, "idx_to_class.json")) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a150a")
    ax.set_facecolor("#0f1f0f")

    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))

    for i in range(NUM_CLASSES):
        y_bin  = (y_true == i).astype(int)
        y_score = y_pred_prob[:, i]

        if y_bin.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc     = auc(fpr, tpr)
        label       = f"{idx_to_class.get(i,'?')[:20]} (AUC={roc_auc:.2f})"
        ax.plot(fpr, tpr, color=colors[i], lw=1.8, label=label)

    ax.plot([0,1],[0,1], "w--", lw=0.8, alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curves — One vs Rest", color="white")
    ax.legend(loc="lower right", fontsize=8, labelcolor="white", framealpha=0.2)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#1a3d1a")
    for spine in ax.spines.values(): spine.set_color("#1a3d1a")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  — "Where is the model looking?"
# This is crucial for a crop disease app: doctors/farmers can verify the
# model is focusing on lesions, not soil or background noise.
# ──────────────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(model, img_array: np.ndarray, pred_index: int = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the given image.

    Grad-CAM: gradient of the score for the top predicted class w.r.t.
    the last convolutional layer activations. High-gradient channels
    correspond to discriminative regions.

    Returns: heatmap (H, W) float32 in [0, 1]
    """
    # Find the last conv layer name inside EfficientNetB0
    base      = model.get_layer("efficientnetb0")
    last_conv = None
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2D layer in the base model.")

    # Create a "feature extractor" that outputs last-conv activations AND predictions
    # We need to go through the entire model, not just the base.
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [base.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array[tf.newaxis, ...], tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the predicted class score w.r.t. last conv output
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over the spatial dimensions (H, W, C) → (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by their gradient importance
    conv_outputs = conv_outputs[0]                             # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]    # (H, W, 1)
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_collage(model, test_paths: list, test_labels: list, n_samples: int = 8):
    """
    Pick n_samples images (2 per class when possible) and save a Grad-CAM grid.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    with open(os.path.join(LOGS_DIR, "idx_to_class.json")) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    # Sample ~2 images from each class
    by_class = {}
    for path, label in zip(test_paths, test_labels):
        by_class.setdefault(label, []).append(path)

    selected = []
    for cls_idx, paths in sorted(by_class.items()):
        for p in paths[:2]:
            selected.append((p, cls_idx))
        if len(selected) >= n_samples:
            break

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, len(selected) * 3.5))
    fig.patch.set_facecolor("#0a150a")

    for row, (img_path, true_cls) in enumerate(selected):
        # Load & preprocess
        orig_bgr = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(orig_rgb, IMAGE_SIZE)

        img_array = preprocess_input(img_resized.astype(np.float32))
        pred_probs = model(img_array[np.newaxis], training=False).numpy()[0]
        pred_cls   = int(np.argmax(pred_probs))
        confidence = pred_probs[pred_cls] * 100

        # Grad-CAM
        heatmap = make_gradcam_heatmap(model, img_array, pred_index=pred_cls)
        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]           # (H, W, 3)
        overlay = (img_resized / 255.0) * 0.6 + heatmap_colored * 0.4 # blend
        overlay = np.clip(overlay, 0, 1)

        true_name = idx_to_class.get(true_cls, str(true_cls))
        pred_name = idx_to_class.get(pred_cls, str(pred_cls))
        correct   = "✓" if true_cls == pred_cls else "✗"

        for col, (img_data, title) in enumerate([
            (img_resized / 255.0, f"True: {true_name[:20]}"),
            (heatmap_resized,     "Grad-CAM Heatmap"),
            (overlay,             f"Pred: {pred_name[:20]} {correct} {confidence:.0f}%"),
        ]):
            ax = axes[row][col]
            ax.imshow(img_data, cmap="jet" if col == 1 else None)
            ax.set_title(title, fontsize=8, color="white")
            ax.axis("off")
            ax.set_facecolor("#0a150a")

    plt.suptitle("Grad-CAM: What is the model looking at?", color="white", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "gradcam_visualization.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM visualization saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def evaluate():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        KrishiAI — Evaluation (Step 3 of 4)              ║")
    print("║   Confusion Matrix · ROC · Grad-CAM · Full Report       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    model, test_ds, test_paths, test_labels = load_model_and_test_data()

    print(f"\nTest samples: {len(test_labels)}")
    print(f"Num classes : {NUM_CLASSES}")

    # 1. Predictions
    y_true, y_pred_cls, y_pred_prob = get_predictions(model, test_ds)

    # 2. Classification report
    accuracy, macro_f1 = print_and_save_report(y_true, y_pred_cls)

    # 3. Confusion matrix
    plot_confusion_matrix(y_true, y_pred_cls)

    # 4. ROC curves
    plot_roc_curves(y_true, y_pred_prob)

    # 5. Grad-CAM
    print("\nGenerating Grad-CAM visualizations...")
    save_gradcam_collage(model, test_paths, test_labels)

    # 6. Save metrics summary
    summary = {
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "num_test_samples": int(len(y_true)),
        "num_classes": NUM_CLASSES,
        "target": ">=90% accuracy for KVK validation",
        "meets_target": bool(accuracy >= 0.90),
    }
    with open(os.path.join(LOGS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Test Accuracy  : {accuracy*100:.2f}%  (target ≥ 90%)")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Target met?    : {'✅ YES' if accuracy >= 0.90 else '⚠️ Not yet — more data or epochs needed'}")
    print(f"\n✅ Evaluation complete!")
    print(f"\nNext step: python step4_export_tflite.py")


if __name__ == "__main__":
    evaluate()
