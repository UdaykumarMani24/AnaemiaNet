"""
05_gradcam_explainability.py  —  Grad-CAM Heatmap Visualisation
────────────────────────────────────────────────────────────────
Generates Grad-CAM heatmaps showing WHAT the model looks at
when predicting anemia from conjunctiva images.

Usage:
    python 05_gradcam_explainability.py --data_dir data/cpanemic_processed
    python 05_gradcam_explainability.py --image path/to/image.jpg
    python 05_gradcam_explainability.py --data_dir data/cpanemic_processed --mode audit

Outputs:
    gradcam_outputs/
        gradcam_grid.png
        gradcam_audit.png
        gradcam_single.png
        per_image/
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras


IMG_SIZE    = 224
CLASS_NAMES = ["anemic", "non_anemic"]
MODEL_DIR   = Path("models")
OUTPUT_DIR  = Path("gradcam_outputs")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(dropout: float = 0.5) -> keras.Model:
    backbone = keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="conjunctiva_input")
    x       = keras.layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)
    x       = backbone(x, training=False)
    x       = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x       = keras.layers.BatchNormalization(name="bn1")(x)
    x       = keras.layers.Dense(256, activation="swish", name="fc1",
                                 kernel_regularizer=keras.regularizers.l2(3e-4))(x)
    x       = keras.layers.Dropout(dropout, name="drop1")(x)
    x       = keras.layers.BatchNormalization(name="bn2")(x)
    x       = keras.layers.Dense(64, activation="swish", name="fc2",
                                 kernel_regularizer=keras.regularizers.l2(3e-4))(x)
    x       = keras.layers.Dropout(dropout / 2, name="drop2")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="anemia_score")(x)
    return keras.Model(inputs, outputs, name="AnaemiaNet_v3")


def load_model() -> keras.Model:
    weight_candidates = [
        "best_model.weights.h5",
        "best_phase2.weights.h5",
        "best_phase1.weights.h5",
        "best_weights.weights.h5",
    ]
    for name in weight_candidates:
        p = MODEL_DIR / name
        if not p.exists():
            continue
        print(f"[✓] Found weights: {p}")
        for unfreeze in (True, False):
            m = build_model()
            m.build((None, IMG_SIZE, IMG_SIZE, 3))
            if unfreeze:
                bb = m.get_layer("efficientnetb0")
                bb.trainable = True
                for layer in bb.layers[:-15]:
                    layer.trainable = False
            try:
                m.load_weights(str(p))
                print(f"[✓] Weights loaded ({'unfrozen' if unfreeze else 'frozen'})")
                return m
            except Exception as e:
                if not unfreeze:
                    print(f"[!] {p.name}: {e}")
    raise FileNotFoundError(
        f"No loadable weights found in '{MODEL_DIR}/'.\n"
        f"Run 02_train_model.py first."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(image_path) -> tuple:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype(np.float32) / 255.0
    return resized, np.expand_dims(norm, axis=0)


def collect_test_images(data_dir: str, n: int = 12) -> list:
    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    samples = []
    for class_name in CLASS_NAMES:
        cls_dir = test_dir / class_name
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append({"path": f, "true_label": class_name})
    random.shuffle(samples)
    return samples[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM  —  single-graph implementation
#
#  Key insight: we build ONE keras.Model that outputs BOTH the target conv
#  feature map AND the final prediction from the SAME input tensor.
#  This keeps everything in a single connected graph so GradientTape
#  can trace gradients all the way through.
# ══════════════════════════════════════════════════════════════════════════════

def _find_backbone_and_conv(model):
    """Extract backbone sub-model and name of its last Conv2D layer."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("EfficientNetB0 sub-model not found.")
    last_conv_name = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    if last_conv_name is None:
        for layer in reversed(backbone.layers):
            if hasattr(layer, "depthwise_kernel"):
                last_conv_name = layer.name
                break
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found in backbone.")
    return backbone, last_conv_name


def build_gradcam_model(model: keras.Model) -> tuple:
    """
    Returns (conv_and_out_model, last_conv_name, head_layers).

    Builds the extractor entirely within EfficientNetB0's own graph to avoid
    the Keras cross-submodel Graph-disconnected error.
    backbone.input -> [conv_features, backbone.output]
    head_layers: AnaemiaNet layers after the backbone (GAP, BN, Dense, ...)
    """
    backbone, last_conv_name = _find_backbone_and_conv(model)
    print(f"[Grad-CAM] Target conv layer: {last_conv_name}")

    conv_and_out = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer(last_conv_name).output,
            backbone.output,
        ],
        name="gradcam_backbone_extractor"
    )

    head_layers = []
    after_backbone = False
    for layer in model.layers:
        if layer is backbone:
            after_backbone = True
            continue
        if after_backbone:
            head_layers.append(layer)

    return conv_and_out, last_conv_name, head_layers


def make_gradcam_heatmap(gradcam_info, img_array: np.ndarray) -> np.ndarray:
    """
    gradcam_info: (conv_and_out_model, last_conv_name, head_layers)
    img_array: (1, 224, 224, 3) float32 in [0, 1]

    The backbone expects [0, 255]; the Rescaling layer in AnaemiaNet did
    that conversion, so we multiply by 255 here before passing to backbone.
    GradientTape watches conv_out; gradients flow through head_layers.
    """
    conv_and_out, _, head_layers = gradcam_info
    img_255 = tf.cast(img_array, tf.float32) * 255.0

    with tf.GradientTape() as tape:
        conv_out, bb_out = conv_and_out(img_255, training=False)
        tape.watch(conv_out)
        x = bb_out
        for layer in head_layers:
            x = layer(x, training=False)
        score = x[:, 0]

    grads = tape.gradient(score, conv_out)

    if grads is None:
        print("[!] Gradient is None - using activation magnitude fallback")
        heatmap = tf.reduce_mean(conv_out[0], axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        hmax = heatmap.max()
        return heatmap / hmax if hmax > 0 else heatmap

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map     = conv_out[0]
    heatmap      = tf.squeeze(tf.nn.relu(conv_map @ pooled_grads[..., tf.newaxis])).numpy()
    hmax         = heatmap.max()
    return heatmap / hmax if hmax > 0 else heatmap


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    h, w       = original_rgb.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_uint8   = np.uint8(255 * hm_resized)
    hm_color   = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_rgb     = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1 - alpha, hm_rgb, alpha, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(model: keras.Model, img_array: np.ndarray) -> tuple:
    prob = float(model.predict(img_array, verbose=0)[0][0])
    if prob >= 0.5:
        return "anemic", prob
    return "non_anemic", 1.0 - prob


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "anemic":     "#E05252",
    "non_anemic": "#4CAF50",
    "wrong":      "#FF9800",
}


def add_result_label(ax, pred_label, confidence, true_label=None):
    correct  = (true_label is None) or (pred_label == true_label)
    color    = PALETTE[pred_label]
    border_c = PALETTE["wrong"] if not correct else color
    display  = pred_label.replace("_", " ").title()
    subtitle = f"{confidence*100:.1f}%"
    if true_label and not correct:
        subtitle += f"\n✗ {true_label.replace('_',' ').title()}"
    ax.set_title(f"{display}\n{subtitle}",
                 fontsize=7, color=border_c, fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_c)
        spine.set_linewidth(2.5)


# ══════════════════════════════════════════════════════════════════════════════
#  GRID MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(model, data_dir, n=12, save_path=None):
    print(f"\n[Grad-CAM Grid] Collecting {n} test images...")
    samples       = collect_test_images(data_dir, n)
    gradcam_info = build_gradcam_model(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_dir = OUTPUT_DIR / "per_image"
    per_dir.mkdir(exist_ok=True)

    cols   = 4
    rows   = (len(samples) + cols - 1) // cols
    fig_w  = cols * 3.0
    fig_h  = rows * 1.8 + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0F0F0F")
    fig.suptitle("AnaemiaNet  ·  Grad-CAM Conjunctiva Attention Maps",
                 fontsize=11, color="white", fontweight="bold", y=0.98)

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['path'].name}")
        original, img_arr = load_image(sample["path"])
        pred_label, conf  = predict(model, img_arr)
        heatmap           = make_gradcam_heatmap(gradcam_info, img_arr)
        overlay           = overlay_heatmap(original, heatmap)

        combined = np.concatenate([original, overlay], axis=1)
        cv2.imwrite(
            str(per_dir / f"{i:02d}_{pred_label}_{sample['path'].stem}.png"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )
        results.append({
            "original":   original, "overlay":    overlay,
            "pred_label": pred_label, "confidence": conf,
            "true_label": sample["true_label"],
        })

    # Layout: 4 image pairs per row → 8 columns (orig|cam repeated 4×)
    pair_cols = 2
    grid_cols = cols * pair_cols
    gs = GridSpec(rows, grid_cols, figure=fig,
                  hspace=0.05, wspace=0.03,
                  left=0.01, right=0.99, top=0.94, bottom=0.04)

    for i, r in enumerate(results):
        row = i // cols
        col = i % cols
        c0  = col * pair_cols
        ax_orig = fig.add_subplot(gs[row, c0])
        ax_cam  = fig.add_subplot(gs[row, c0 + 1])

        ax_orig.imshow(r["original"]); ax_orig.axis("off")
        ax_orig.set_title("Original", fontsize=5, color="#888", pad=2)
        ax_cam.imshow(r["overlay"]);   ax_cam.axis("off")
        add_result_label(ax_cam, r["pred_label"], r["confidence"], r["true_label"])

    patches = [
        mpatches.Patch(color=PALETTE["anemic"],     label="Anemic"),
        mpatches.Patch(color=PALETTE["non_anemic"], label="Non-Anemic"),
        mpatches.Patch(color=PALETTE["wrong"],      label="Wrong prediction"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.2, labelcolor="white", facecolor="#1A1A1A",
               bbox_to_anchor=(0.5, 0.0))

    out = save_path or (OUTPUT_DIR / "gradcam_grid.png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"\n[✓] Grid saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_audit(model, data_dir):
    print("\n[Grad-CAM Audit] Loading all test images...")
    all_samples      = collect_test_images(data_dir, n=200)
    gradcam_info = build_gradcam_model(model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, sample in enumerate(all_samples):
        original, img_arr = load_image(sample["path"])
        pred_label, conf  = predict(model, img_arr)
        true_label        = sample["true_label"]
        correct           = pred_label == true_label
        heatmap           = make_gradcam_heatmap(gradcam_info, img_arr)
        overlay           = overlay_heatmap(original, heatmap)
        all_results.append({
            "original": original, "overlay": overlay,
            "pred_label": pred_label, "confidence": conf,
            "true_label": true_label, "correct": correct,
        })
        print(f"  [{i+1}/{len(all_samples)}] {'✓' if correct else '✗'} "
              f"{pred_label} ({conf*100:.1f}%)")

    correct_top = sorted(
        [r for r in all_results if r["correct"]],
        key=lambda x: x["confidence"], reverse=True)[:5]
    error_top = sorted(
        [r for r in all_results if not r["correct"]],
        key=lambda x: x["confidence"], reverse=True)[:5]

    n_cols = max(len(correct_top), len(error_top), 1)
    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.5, 10),
                              facecolor="#0F0F0F")
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Grad-CAM Audit: Most Confident Correct vs Worst Errors",
                 fontsize=11, color="white", fontweight="bold")

    for j, r in enumerate(correct_top):
        axes[0, j].imshow(r["original"]); axes[0, j].axis("off")
        axes[1, j].imshow(r["overlay"]);  axes[1, j].axis("off")
        add_result_label(axes[1, j], r["pred_label"], r["confidence"], r["true_label"])
    for j in range(len(correct_top), n_cols):
        axes[0, j].axis("off"); axes[1, j].axis("off")

    for j, r in enumerate(error_top):
        axes[2, j].imshow(r["original"]); axes[2, j].axis("off")
        axes[3, j].imshow(r["overlay"]);  axes[3, j].axis("off")
        add_result_label(axes[3, j], r["pred_label"], r["confidence"], r["true_label"])
    for j in range(len(error_top), n_cols):
        axes[2, j].axis("off"); axes[3, j].axis("off")

    for ax in axes.flat:
        ax.set_facecolor("#0F0F0F")

    for row_idx, label in enumerate(["Original", "Correct ✓", "Original", "Error ✗"]):
        axes[row_idx, 0].set_ylabel(label, color="white", fontsize=8,
                                    rotation=0, ha="right", labelpad=40)

    out = OUTPUT_DIR / "gradcam_audit.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"\n[✓] Audit saved → {out}")

    n_correct = sum(1 for r in all_results if r["correct"])
    print(f"\nTest accuracy: {n_correct}/{len(all_results)} "
          f"({n_correct/len(all_results)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_single(model, image_path):
    gradcam_info = build_gradcam_model(model)
    original, img_arr = load_image(image_path)
    pred_label, conf  = predict(model, img_arr)
    heatmap           = make_gradcam_heatmap(gradcam_info, img_arr)
    overlay           = overlay_heatmap(original, heatmap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), facecolor="#0F0F0F")

    axes[0].imshow(original)
    axes[0].set_title("Original", color="white", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay", color="white", fontsize=9)
    axes[1].axis("off")
    add_result_label(axes[1], pred_label, conf)

    axes[2].imshow(heatmap, cmap="jet")
    axes[2].set_title("Raw Heatmap", color="white", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(f"AnaemiaNet  ·  {Path(image_path).name}",
                 color="white", fontsize=10, fontweight="bold")
    out = OUTPUT_DIR / "gradcam_single.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"\n[✓] {pred_label.replace('_',' ').title()} ({conf*100:.1f}%)")
    print(f"[✓] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM explainability for AnaemiaNet"
    )
    parser.add_argument("--data_dir", default="data/cpanemic_processed")
    parser.add_argument("--image",    default=None)
    parser.add_argument("--mode",     choices=["grid", "audit"], default="grid")
    parser.add_argument("--n",        type=int, default=12)
    args = parser.parse_args()

    print("=" * 60)
    print("  AnaemiaNet  ·  Grad-CAM Explainability")
    print("=" * 60)

    model = load_model()

    if args.image:
        print(f"\n[Mode] Single image: {args.image}")
        run_single(model, args.image)
    elif args.mode == "audit":
        print(f"\n[Mode] Audit")
        run_audit(model, args.data_dir)
    else:
        print(f"\n[Mode] Grid ({args.n} images)")
        run_grid(model, args.data_dir, n=args.n)

    print("\n[Done] All outputs in gradcam_outputs/")


if __name__ == "__main__":
    main()
