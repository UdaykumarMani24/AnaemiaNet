"""
02_train_model.py  —  AnaemiaNet Training Pipeline (v3)
─────────────────────────────────────────────────────────
Fixes: EagerTensor JSON crash caused by (inputs * 255.0) in model graph
       + class_weight injecting tensor array into Keras logs dict.

Solution:
  1. Use keras.layers.Rescaling(255.0) instead of inputs * 255.0
     → no TFOpLambda in graph, model serializes cleanly
  2. Use sample_weight instead of class_weight
     → no EagerTensor injected into logs dict
  3. No CSVLogger, No TensorBoard — replaced with SafeCSVLogger

Usage
─────
    python 02_train_model.py --data_dir data\cpanemic_processed
"""

import argparse
import sys
import json
import math
import time
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["anemic", "non_anemic"]
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        type=Path,  required=True)
    p.add_argument("--output_dir",      type=Path,  default=Path("models"))
    p.add_argument("--img_size",        type=int,   default=224)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--epochs_phase1",   type=int,   default=25)
    p.add_argument("--epochs_phase2",   type=int,   default=40)
    p.add_argument("--lr_phase1",       type=float, default=1e-3)
    p.add_argument("--lr_phase2",       type=float, default=5e-5)
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--finetune_layers", type=int,   default=15)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--no_augment",      action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
#  Uses sample_weight instead of class_weight to avoid EagerTensor in logs
# ══════════════════════════════════════════════════════════════════════════════

def compute_sample_weights(labels: np.ndarray, class_weights: dict) -> np.ndarray:
    """Convert class_weight dict → per-sample weight array."""
    return np.array([class_weights[int(l)] for l in labels], dtype=np.float32)


def augment_fn(image, label, weight):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.88, upper=1.12)
    pad  = 24
    size = tf.shape(image)[0]
    image = tf.image.resize_with_crop_or_pad(image, size + pad, size + pad)
    image = tf.image.random_crop(image, [size, size, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label, weight


def load_split(data_dir: Path, split: str, img_size: int,
               batch_size: int, class_weights: dict = None,
               augment: bool = False) -> tf.data.Dataset:
    split_dir = data_dir / split
    if not split_dir.exists():
        sys.exit(f"[ERROR] Not found: {split_dir}")

    # Collect all file paths + labels
    all_files, all_labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for f in sorted(cls_dir.glob(ext)):
                all_files.append(str(f))
                all_labels.append(float(cls_idx))

    all_labels  = np.array(all_labels, dtype=np.float32)

    # Sample weights — plain numpy floats, never become EagerTensors in logs
    if class_weights:
        sample_w = compute_sample_weights(all_labels, class_weights)
    else:
        sample_w = np.ones(len(all_labels), dtype=np.float32)

    def parse_image(path, label, weight):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label, weight

    ds = tf.data.Dataset.from_tensor_slices(
        (all_files, all_labels, sample_w)
    )

    if split == "train":
        ds = ds.shuffle(buffer_size=len(all_files), seed=SEED)

    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def get_class_weights(data_dir: Path) -> dict:
    train_dir = data_dir / "train"
    labels = []
    for idx, cls in enumerate(CLASS_NAMES):
        n = len(list((train_dir / cls).glob("*.*")))
        labels.extend([idx] * n)
    y  = np.array(labels)
    w  = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cw = {int(k): float(v) for k, v in zip(np.unique(y), w)}
    print(f"  class_weight = {cw}")
    return cw


def print_split_info(data_dir: Path):
    print("\n[Dataset] CP-AnemiC split summary:")
    for split in ("train", "val", "test"):
        row = []
        for cls in CLASS_NAMES:
            d = data_dir / split / cls
            n = len(list(d.glob("*.*"))) if d.exists() else 0
            row.append(f"{cls}={n}")
        print(f"  {split:5s}: {',  '.join(row)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
#  Uses keras.layers.Rescaling instead of (inputs * 255.0)
#  Rescaling is a proper Keras layer — it serializes to JSON without error.
# ══════════════════════════════════════════════════════════════════════════════

def build_model(img_size: int, dropout: float) -> keras.Model:
    backbone = keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3), name="conjunctiva_input")

    # ── KEY FIX: Rescaling layer instead of (inputs * 255.0) ──────────────
    # (inputs * 255.0) creates a TFOpLambda node that stores the constant
    # 255.0 as an EagerTensor in the model config, breaking json.dumps().
    # keras.layers.Rescaling is a proper registered layer with clean JSON.
    x = layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dense(256, activation="swish", name="fc1",
                     kernel_regularizer=keras.regularizers.l2(3e-4))(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dense(64, activation="swish", name="fc2",
                     kernel_regularizer=keras.regularizers.l2(3e-4))(x)
    x = layers.Dropout(dropout / 2, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anemia_score")(x)

    return keras.Model(inputs, outputs, name="AnaemiaNet_v3")


def unfreeze_top_layers(model: keras.Model, n_layers: int = 15):
    backbone  = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    trainable = sum(1 for l in backbone.layers if l.trainable)
    frozen    = sum(1 for l in backbone.layers if not l.trainable)
    print(f"  EfficientNetB0: {trainable} unfrozen, {frozen} frozen")


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE CSV LOGGER  (never calls json.dumps)
# ══════════════════════════════════════════════════════════════════════════════

class SafeCSVLogger(keras.callbacks.Callback):
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self._file    = None
        self._writer  = None

    @staticmethod
    def _safe(val):
        if hasattr(val, "numpy"):
            val = val.numpy()
        if hasattr(val, "flat"):
            return float(list(val.flat)[0])
        if hasattr(val, "item"):
            return val.item()
        try:
            return float(val)
        except Exception:
            return str(val)

    def on_train_begin(self, logs=None):
        self._file   = open(self.filepath, "w", newline="")
        self._writer = None

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch}
        for k, v in (logs or {}).items():
            row[k] = self._safe(v)
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, logs=None):
        if self._file:
            self._file.close()


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class SafeCheckpoint(keras.callbacks.Callback):
    """
    Replacement for ModelCheckpoint.
    Saves only the WEIGHTS (not the full model config) using save_weights(),
    which writes an HDF5 file and never calls json.dumps() on the model graph.
    This avoids the EagerTensor serialization crash that occurs when
    EfficientNetB0's internal Rescaling/BatchNorm layers store their
    parameters as EagerTensors in the model JSON config.
    """
    def __init__(self, filepath: str, monitor: str = "val_auc", mode: str = "max"):
        super().__init__()
        self.filepath  = filepath
        self.monitor   = monitor
        self.mode      = mode
        self.best      = -np.inf if mode == "max" else np.inf

    def _is_better(self, current):
        return current > self.best if self.mode == "max" else current < self.best

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        val     = logs.get(self.monitor)
        if val is None:
            return
        # Convert EagerTensor → float safely
        if hasattr(val, "numpy"):
            val = float(val.numpy().flat[0])
        else:
            val = float(val)
        if self._is_better(val):
            self.best = val
            self.model.save_weights(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {val:.5f} "
                  f"→ saved weights to {self.filepath}")


def make_callbacks(output_dir: Path, phase: int, patience: int,
                   lr: float, total_epochs: int):
    # Weights-only checkpoint (.weights.h5) — never serializes model JSON
    ckpt_path = str(output_dir / f"best_phase{phase}.weights.h5")
    min_lr    = lr * 0.01

    def cosine_schedule(epoch, _lr):
        progress = epoch / max(total_epochs - 1, 1)
        return float(min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress)))

    return [
        SafeCheckpoint(ckpt_path, monitor="val_auc", mode="max"),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        SafeCSVLogger(str(output_dir / f"training_phase{phase}.csv")),
        keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=0),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_ds, output_dir: Path):
    print("\n[Evaluation] Running on test set …")
    y_true, y_prob = [], []
    for images, labels, _ in test_ds:
        probs = model.predict(images, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores    = tpr - fpr
    best_idx    = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    print(f"  Optimal threshold (Youden's J): {best_thresh:.3f}")

    y_pred = (y_prob >= best_thresh).astype(int)
    auc    = roc_auc_score(y_true, y_prob)

    print(f"\n  AUC-ROC : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    metrics = {
        "auc_roc": round(auc, 4), "threshold": round(best_thresh, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.scatter(fpr[best_idx], tpr[best_idx], color="navy", zorder=5, s=80,
                label=f"Thresh={best_thresh:.2f} Sn={sensitivity:.2f} Sp={specificity:.2f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("AnaemiaNet v3 — ROC Curve"); plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(output_dir / "roc_curve.png", dpi=150); plt.close()

    # History plots
    for phase in (1, 2):
        csv_path = output_dir / f"training_phase{phase}.csv"
        if not csv_path.exists():
            continue
        rows   = list(csv.DictReader(open(csv_path)))
        epochs = [int(r["epoch"]) + 1 for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, [r["loss"]     for r in rows], label="Train loss")
        ax1.plot(epochs, [r["val_loss"] for r in rows], label="Val loss")
        ax1.set_title(f"Phase {phase} — Loss"); ax1.legend()
        ax2.plot(epochs, [r.get("auc","")     for r in rows], label="Train AUC")
        ax2.plot(epochs, [r.get("val_auc","") for r in rows], label="Val AUC")
        ax2.set_title(f"Phase {phase} — AUC"); ax2.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"history_phase{phase}.png", dpi=150); plt.close()

    print(f"  Reports saved → {output_dir}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  TFLITE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_tflite(model, output_dir: Path, test_ds):
    print("\n[Export] FP16 TFLite …")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    fp16_bytes = conv.convert()
    fp16_path  = output_dir / "anaemia_model_fp16.tflite"
    fp16_path.write_bytes(fp16_bytes)
    print(f"  Saved: {fp16_path}  ({len(fp16_bytes)/1024:.1f} KB)")

    print("[Export] INT8 TFLite …")
    rep_images = []
    for imgs, _, _ in test_ds.take(7):
        rep_images.append(imgs.numpy().astype(np.float32))
    rep_images = np.concatenate(rep_images, axis=0)[:100]

    def representative_dataset():
        for img in rep_images:
            yield [img[np.newaxis, ...]]

    conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv2.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv2.representative_dataset    = representative_dataset
    conv2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv2.inference_input_type      = tf.uint8
    conv2.inference_output_type     = tf.uint8
    int8_bytes = conv2.convert()
    int8_path  = output_dir / "anaemia_model_int8.tflite"
    int8_path.write_bytes(int8_bytes)
    print(f"  Saved: {int8_path}  ({len(int8_bytes)/1024:.1f} KB)")

    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(CLASS_NAMES))
    print(f"  Saved: {labels_path}")

    assets_dir = Path("android_app/app/src/main/assets")
    if assets_dir.exists():
        import shutil
        shutil.copy2(fp16_path,   assets_dir / "anaemia_model_fp16.tflite")
        shutil.copy2(labels_path, assets_dir / "labels.txt")
        print(f"  Copied to Android assets: {assets_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AnaemiaNet v3 Training  —  CP-AnemiC Dataset")
    print("=" * 60)
    print(f"  backbone       : EfficientNetB0")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  img_size       : {args.img_size}×{args.img_size}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  dropout        : {args.dropout}")
    print(f"  finetune_layers: {args.finetune_layers}")
    print(f"  Phase-1        : {args.epochs_phase1} epochs  lr={args.lr_phase1}")
    print(f"  Phase-2        : {args.epochs_phase2} epochs  lr={args.lr_phase2}")
    gpu = tf.config.list_physical_devices("GPU")
    print(f"  GPU            : {gpu[0].name if gpu else 'None (CPU)'}")
    print()

    print_split_info(args.data_dir)

    print("[Dataset] Computing class weights …")
    cw = get_class_weights(args.data_dir)

    # Load with sample_weight baked into dataset (no class_weight= in fit())
    train_ds = load_split(args.data_dir, "train", args.img_size, args.batch_size,
                          class_weights=cw, augment=not args.no_augment)
    val_ds   = load_split(args.data_dir, "val",   args.img_size, args.batch_size)
    test_ds  = load_split(args.data_dir, "test",  args.img_size, args.batch_size)

    print("\n[Model] Building AnaemiaNet v3 …")
    model = build_model(args.img_size, args.dropout)
    model.summary(line_length=80)

    # ── PHASE 1 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — Head training  ({args.epochs_phase1} epochs max)")
    print(f"  Backbone frozen")
    print(f"{'─'*60}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_phase1),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    t0 = time.time()
    model.fit(
        train_ds,          # dataset yields (image, label, sample_weight)
        validation_data=val_ds,
        epochs=args.epochs_phase1,
        # NO class_weight= here — weights are baked into the dataset
        callbacks=make_callbacks(args.output_dir, phase=1,
                                 patience=args.patience,
                                 lr=args.lr_phase1,
                                 total_epochs=args.epochs_phase1),
        verbose=1,
    )
    print(f"  Phase-1 done in {(time.time()-t0)/60:.1f} min")

    # ── PHASE 2 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — Fine-tuning top {args.finetune_layers} layers "
          f"({args.epochs_phase2} epochs max)")
    print(f"{'─'*60}")

    unfreeze_top_layers(model, n_layers=args.finetune_layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_phase2),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase2,
        callbacks=make_callbacks(args.output_dir, phase=2,
                                 patience=args.patience,
                                 lr=args.lr_phase2,
                                 total_epochs=args.epochs_phase2),
        verbose=1,
    )
    print(f"  Phase-2 done in {(time.time()-t0)/60:.1f} min")

    # Save weights only — model.save() crashes on this TF version because
    # EfficientNetB0 stores EagerTensors in its JSON config.
    weights_path = str(args.output_dir / "best_model.weights.h5")
    model.save_weights(weights_path)
    print(f"\n[Save] Weights saved → {weights_path}")

    # Rebuild a clean model with no training-time EagerTensors,
    # load the saved weights, then evaluate and export.
    print("[Rebuild] Loading best weights into fresh model ...")
    export_model = build_model(args.img_size, args.dropout)
    unfreeze_top_layers(export_model, n_layers=args.finetune_layers)
    export_model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(),
    )
    export_model.load_weights(weights_path)
    print("[Rebuild] Done.")

    metrics = evaluate_model(export_model, test_ds, args.output_dir)
    export_tflite(export_model, args.output_dir, test_ds)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)
    print(f"  AUC-ROC     : {metrics['auc_roc']}")
    print(f"  Sensitivity : {metrics['sensitivity']}")
    print(f"  Specificity : {metrics['specificity']}")
    print(f"\n  Key outputs in {args.output_dir}/")
    print("    best_model.keras")
    print("    anaemia_model_fp16.tflite   ← copy to Android assets/")
    print("    anaemia_model_int8.tflite")
    print("    labels.txt")
    print("    roc_curve.png")
    print("    test_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
