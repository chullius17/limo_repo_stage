#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_PATH = "best.onnx"
IMAGE_PATH = "example_img.jpg"

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_PATH}")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Outputs: {[o.name for o in session.get_outputs()]}\n")

# --- LOAD & PREPROCESS IMAGE ---
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print(f"Original image shape: {img_bgr.shape}")

input_img = cv2.resize(img_bgr, (448, 448))
input_img = input_img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0  # (1,3,448,448)

# --- INFERENCE ---
outputs = session.run(None, {input_name: input_tensor})

# --- PRINT RAW OUTPUT INFO ---
for i, out in enumerate(outputs):
    print(f"Output[{i}] — shape: {out.shape}, dtype: {out.dtype}")
    print(f"  min={out.min():.4f}  max={out.max():.4f}  mean={out.mean():.4f}")
    print()

# Detailed look at predictions (output[0]) after transpose
predictions = np.squeeze(outputs[0]).T   # (N, 4+num_classes+32)
print(f"Predictions matrix shape (after squeeze+T): {predictions.shape}")

num_classes = 4
scores = predictions[:, 4:4+num_classes]
class_ids = np.argmax(scores, axis=1)
confidences = scores[np.arange(len(predictions)), class_ids]

print(f"\nConfidence stats across all {len(predictions)} anchors:")
print(f"  max={confidences.max():.4f}  mean={confidences.mean():.4f}")

threshold = 0.1
kept = predictions[confidences > threshold]
kept_class_ids = class_ids[confidences > threshold]
kept_confs = confidences[confidences > threshold]

print(f"\nDetections above threshold ({threshold}): {len(kept)}")
if len(kept) == 0:
    print("No detections found — try lowering the threshold.")
    exit(0)

for cls in range(num_classes):
    n = np.sum(kept_class_ids == cls)
    print(f"  Class {cls}: {n} detections")

# --- PICK the detection with the LARGEST bounding box among all above threshold ---
# boxes are (cx, cy, bw, bh) in pixel space at 448x448
areas = kept[:, 2] * kept[:, 3]   # bw * bh
best_idx = np.argmax(areas)
best_pred = kept[best_idx]
best_class = kept_class_ids[best_idx]
best_conf = kept_confs[best_idx]

cx, cy, bw, bh = best_pred[:4]
x1 = cx - bw / 2
y1 = cy - bh / 2
print(f"\nLargest bbox among all detections above threshold:")
print(f"  class={best_class}  conf={best_conf:.4f}  cx={cx:.1f} cy={cy:.1f} bw={bw:.1f} bh={bh:.1f}")

# Scale box back to original image size
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
oh, ow = img_rgb.shape[:2]
sx, sy = ow / 448.0, oh / 448.0

CLASS_NAMES = {0: "Dashed", 1: "Parking", 2: "Solid", 3: "Road"}
CLASS_COLORS = {0: "lime", 1: "dodgerblue", 2: "red", 3: "gray"}

# ── PLOT 1: Image + bounding box ──────────────────────────────────────────────
fig1, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.imshow(img_rgb)
rect = patches.Rectangle(
    (x1 * sx, y1 * sy), bw * sx, bh * sy,
    linewidth=2,
    edgecolor=CLASS_COLORS.get(best_class, "yellow"),
    facecolor="none"
)
ax.add_patch(rect)
ax.set_title(
    f"Largest bbox (conf > {threshold})  |  class: {CLASS_NAMES.get(best_class, best_class)}  |  conf: {best_conf:.3f}",
    fontsize=12
)
ax.axis("off")
plt.tight_layout()
plt.savefig("plot_bbox.png", dpi=150)
print("\nSaved: plot_bbox.png")

# ── PLOT 2: Histogram of the 32 mask coefficients for the best detection ──────
mask_coeffs = best_pred[4 + num_classes : 4 + num_classes + 32]  # (32,)

fig2, ax2 = plt.subplots(figsize=(12, 4))
colors = ["tomato" if v < 0 else "steelblue" for v in mask_coeffs]
ax2.bar(np.arange(32), mask_coeffs, color=colors, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xticks(np.arange(32))
ax2.set_xticklabels([str(i) for i in range(32)], fontsize=8)
ax2.set_xlabel("Prototype index")
ax2.set_ylabel("Coefficient value")
ax2.set_title(
    f"Mask coefficients — largest bbox  |  class: {CLASS_NAMES.get(best_class, best_class)}  |  conf: {best_conf:.3f}\n"
    f"(blue = positive contribution, red = negative)",
    fontsize=11
)
plt.tight_layout()
plt.savefig("plot_coefficients.png", dpi=150)
print("Saved: plot_coefficients.png")

# ── PLOT 3: 32 prototype masks (4 rows × 8 cols) ─────────────────────────────
proto = np.squeeze(outputs[1])   # (32, 112, 112)
assert proto.shape[0] == 32, f"Expected 32 prototypes, got {proto.shape[0]}"

fig3, axes3 = plt.subplots(4, 8, figsize=(18, 9))
fig3.suptitle("32 Prototype Masks  (raw, before sigmoid)", fontsize=14, y=1.01)

for i, ax3 in enumerate(axes3.flat):
    mask = proto[i]                             # (112, 112)
    ax3.imshow(mask, cmap="gray", interpolation="nearest")
    ax3.set_title(f"proto {i}", fontsize=7)
    ax3.axis("off")

plt.tight_layout()
plt.savefig("plot_prototypes.png", dpi=150, bbox_inches="tight")
print("Saved: plot_prototypes.png")

# ── PLOT 4: Combined mask overlay on original image ───────────────────────────
# 1. Linear combination: coeffs @ proto  →  (112, 112) logit map
proto_flat = proto.reshape(32, -1)                          # (32, 112*112)
raw_mask = (mask_coeffs @ proto_flat).reshape(112, 112)     # (112, 112)

# 2. Crop to bounding box at proto scale (112/448 = 0.25)
scale_proto = 112.0 / 448.0
px1 = int(np.clip((cx - bw / 2) * scale_proto, 0, 111))
py1 = int(np.clip((cy - bh / 2) * scale_proto, 0, 111))
px2 = int(np.clip((cx + bw / 2) * scale_proto, 0, 111))
py2 = int(np.clip((cy + bh / 2) * scale_proto, 0, 111))

cropped_logit = np.full((112, 112), -10.0, dtype=np.float32)
cropped_logit[py1:py2, px1:px2] = raw_mask[py1:py2, px1:px2]

# 3. Sigmoid → binary mask at proto resolution
prob_map = 1.0 / (1.0 + np.exp(-cropped_logit))            # (112, 112)
binary_low = (prob_map > 0.5).astype(np.uint8)

# 4. Upsample to original image size
binary_full = cv2.resize(binary_low, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)

# 5. Colour overlay
color_bgr = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0)}
overlay = np.zeros_like(img_bgr)
overlay[binary_full] = color_bgr.get(best_class, (0, 255, 255))

blended = cv2.addWeighted(img_bgr, 1.0, overlay, 0.5, 0)
blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.imshow(blended_rgb)
ax4.set_title(
    f"Segmentation mask — class: {CLASS_NAMES.get(best_class, best_class)}  |  conf: {best_conf:.3f}",
    fontsize=12
)
ax4.axis("off")
plt.tight_layout()
plt.savefig("plot_mask_overlay.png", dpi=150)
print("Saved: plot_mask_overlay.png")

plt.show()