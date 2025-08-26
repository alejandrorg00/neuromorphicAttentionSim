# -*- coding: utf-8 -*-
import os, pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# ===================== Paths & IDs =====================
object_id = "cofeecup_5s_1000fps_720x480"

base_dir  = os.path.join("../attention/objectData", object_id)
fix_path  = os.path.join(base_dir, "fixational_xy.pkl")
rois_path = os.path.join(base_dir, "ROIs.pkl")

render_video_path = os.path.join("../blenderFrame/renders", f"{object_id}.mp4")
dvs_video_path    = os.path.join("../eventFrame/outputs", f"{object_id}.avi")
npy_path          = os.path.join("../eventFrame/outputs", f"{object_id}.npy")

# ===================== Load fixation path & ROI crops =====================
with open(fix_path, "rb") as f:
    attention_xy = np.asarray(pkl.load(f))   # shape [N, 2] (x, y)

with open(rois_path, "rb") as f:
    crops_bgr = pkl.load(f)                  # list of ROI crops (BGR images)
if len(crops_bgr) == 0:
    raise ValueError("ROIs.pkl is empty.")

# ===================== First RGB frame from render =====================
first_frame_rgb = None
if os.path.isfile(render_video_path):
    cap = cv2.VideoCapture(render_video_path)
    ret, frame_bgr = cap.read()
    cap.release()
    if ret:
        first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
if first_frame_rgb is None:
    # fallback: use first ROI crop if render not available
    first_frame_rgb = cv2.cvtColor(crops_bgr[0], cv2.COLOR_BGR2RGB)

H, W = first_frame_rgb.shape[:2]

# ===================== First DVS frame from .avi =====================
cap = cv2.VideoCapture(dvs_video_path)
ret, dvs_bgr = cap.read()
cap.release()
if not ret:
    raise IOError(f"Could not read first frame from {dvs_video_path}")
dvs_gray_avi = cv2.cvtColor(dvs_bgr, cv2.COLOR_BGR2GRAY)

# ===================== First DVS frame from .npy =====================
data = np.load(npy_path)              # columns: [x, y, polarity, t]
x = data[:, 0].astype(int)
y = data[:, 1].astype(int)
t = data[:, 3] * 1e3                  # convert to ms

WDVS = x.max() + 1
HDVS = y.max() + 1

window_ms = 2.0                       # event window
t0 = t.min()
mask = (t <= t0 + window_ms)

dvs_first = np.zeros((HDVS, WDVS), dtype=np.uint8)
dvs_first[y[mask], x[mask]] = 255

# ===================== ROI selection =====================
# indices are 1-based for labels, crops are 0-based
roi_indices = [1, 3, 39, 68]

roi_labels = [rf'$\mathrm{{ROI}}_{{{i}}}$' for i in roi_indices]
selected_coords = attention_xy[roi_indices]       
selected_crops  = [crops_bgr[i-1] for i in roi_indices]

# crop rectangle size from first ROI
ch, cw = selected_crops[0].shape[:2]
half_w, half_h = cw // 2, ch // 2
crop_w, crop_h = 2 * half_w, 2 * half_h

# ===================== Build figure (2×2) =====================
fig = plt.figure(figsize=(13, 12))
outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.025)

# (0,0): RGB frame
ax1 = fig.add_subplot(outer[0, 0])
ax1.imshow(first_frame_rgb)
ax1.axis('off')
ax1.set_title("Setup (RGB frame)", fontweight='bold', fontsize=18, pad=20)

# (0,1): DVS frame from .avi
ax2 = fig.add_subplot(outer[0, 1])
ax2.imshow(dvs_gray_avi, cmap='viridis', vmin=0, vmax=255)
ax2.axis('off')
ax2.set_title("DVS events", fontweight='bold', fontsize=18, pad=20)

# (1,0): DVS from .npy + path + ROIs
ax3 = fig.add_subplot(outer[1, 0])
ax3.imshow(dvs_first, cmap='binary', vmin=0, vmax=255)

# saccadic path
ax3.plot(attention_xy[:, 0], attention_xy[:, 1],
         '-', linewidth=2, alpha=0.85, color='royalblue')
ax3.scatter(attention_xy[:, 0], attention_xy[:, 1],
            s=20, edgecolors='blue', facecolors='none', linewidth=1.5)

# ROI rectangles and labels
for (x0, y0), label in zip(selected_coords, roi_labels):
    x = int(np.clip(x0, 0, WDVS - 1))
    y = int(np.clip(y0, 0, HDVS - 1))
    rect = plt.Rectangle((x - half_w, y - half_h), crop_w, crop_h,
                         edgecolor='crimson', facecolor='none', linewidth=2)
    ax3.add_patch(rect)
    ax3.text(x, y + max(60, half_h//2), label, color='crimson',
             fontsize=14, ha='center', fontweight='bold')

ax3.set_aspect('equal', 'box')
ax3.set_xlim(0, WDVS - 1)
ax3.set_ylim(HDVS - 1, 0)
ax3.set_xlabel(f'x ({WDVS} px)', fontsize=16)
ax3.set_ylabel(f'y ({HDVS} px)', fontsize=16)
ax3.set_title(
    r"$\mathbf{Saccades \, and \, fixations}$" "\n"
    r"$\{ (\Delta x_1,\Delta y_1), \dots, (\Delta x_n,\Delta y_n) \}$",
    fontweight='bold', color='royalblue', pad=30, fontsize=18
)

# (1,1): ROI crops (2×2 grid)
ax4 = fig.add_subplot(outer[1, 1])
ax4.axis('off')
ax4.set_title(
    r"$\mathbf{Regions\ of\ Interest\ (ROIs)}$" "\n"
    r"$\{\ ROI_1,\dots,\ ROI_{n}\ \}$",
    fontweight='bold', color='crimson', pad=0, fontsize=18, y=0.94)

inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 1],
                                         wspace=0.02, hspace=-0.4)

for idx, (img_bgr, label) in enumerate(zip(selected_crops, roi_labels)):
    i, j = divmod(idx, 2)
    ax = fig.add_subplot(inner[i, j])
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.text(0.1, 0.92, label,
            transform=ax.transAxes,
            fontsize=13,
            color='crimson',
            fontweight='bold',
            ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.25'))

plt.show()

# Save
fig.savefig('pipeline_fig2x2.svg', format='svg', dpi=600, bbox_inches='tight')
fig.savefig('pipeline_fig2x2.pdf', format='pdf', dpi=600, bbox_inches='tight')
