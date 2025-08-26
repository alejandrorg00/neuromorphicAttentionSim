# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:37:30 2025
@author: c3055922

    Adapted from CTU-EDNeuromorphic tutorial (Giulia D'Angelo, 
                                              giulia.dangelo@fel.cvut.cz)
    
    This script implements a saliency-based attention mechanism using 
    event-driven data.The attention mechanism is designed to highlight the most
    significant features of the scene, using configurable parameters that 
    define the characteristics of the attention arcs and kernels.The script 
    leverages PyTorch for efficient tensor operations and OpenCV for real-time 
    visualization of the saliency maps.
    
    This implementation aims to demonstrate how attention mechanisms can be 
    applied to dynamic visual data, enabling robots and systems to focus on 
    relevant features in real-time.
    
"""
import os
import numpy as np
from helpers.helpers import initialise_attention, run_attention
import torch
import cv2
import matplotlib.pyplot as plt
import pickle as pkl


# Configuration class to store attention parameters
class Config:
    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn': 16,  # Size of the kernel used in the attention mechanism
        'r0': 14,  # Radius shift from the center for the attention arc
        'rho': 0.05,  # Scale coefficient to control the arc length
        'theta': np.pi * 3 / 2,  # Angle to control the orientation of the arc
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),  # Array of angles for multi-directional attention
        'thick': 3,  # Thickness of the arc in the attention map
        'fltr_resize_perc': [2, 2],  # Resize percentage for filters
        'offsetpxs': 0,  # Offset in pixels (half the size of the kernel)
        'offset': (0, 0),  # Offset for attention positioning
        'num_pyr': 6,  # Number of pyramid levels in the attention network
        'tau_mem': 0.3,  # Memory time constant for the attention mechanism
        'stride': 1,  # Stride for attention processing
        'out_ch': 1  # Number of output channels
    }


# Set the device for PyTorch (using Metal Performance Shaders on macOS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the configuration
config = Config()

# Load event data from a .npy file containing two objects
data_path = '../eventFrame/outputs/cofeecup_5s_1000fps_720x480.npy'
data = np.load(data_path)
object_id = os.path.splitext(os.path.basename(data_path))[0]

# Extract coordinates and properties from the data
x, y, p, t = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2], data[:, 3] * 1e3

# Determine the resolution based on the maximum coordinates
max_x = x.max() + 1  # Maximum x coordinate + 1 for resolution
max_y = y.max() + 1  # Maximum y coordinate + 1 for resolution
resolution = (max_y, max_x)  # Resolution tuple for attention processing

# Initialize saliency map and coordinates for maximum saliency
saliency_map = np.zeros((max_y, max_x), dtype=np.float32)  # Saliency map initialized to zero
salmax_coords = np.zeros((2,), dtype=np.int32)  # Array to hold coordinates of maximum saliency

##### Attention Mechanism #####
# Initialize the attention modules with the specified device and parameters
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

# Set the time window period for processing events (in milliseconds)
window_period = 2  # Time window in milliseconds
time = window_period  # Initialize the time variable
window = torch.zeros((1, max_y, max_x), dtype=torch.float32)  # Create a tensor to hold the current window of events

### OBJECT IDENTIFICATION (ROIs and fixational points)
# Attention storage
attention_xy = [(max_x/2, max_y/2)] # initialize with attention in the middle

# Fovea fixation
per = 0.1 # 10% fovea
crop_x = max(1, int(max_x * per))
half_x = crop_x // 2
crop_y = max(1, int(max_y * per))
half_y = crop_y // 2
crops = [] # raw crops


# Iterate through the event data
for xi, yi, pi, ti in zip(x, y, p, t):
    if ti <= time:
        # If the event time is within the current time window, update the window
        window[0][yi][xi] = 255  # Mark the pixel corresponding to the event
    else:
        # If the event time exceeds the current time window, process the attention
        saliency_map[:], salmax_coords[:] = run_attention(window, net_attention, device, resolution,
                                                          config.ATTENTION_PARAMS['num_pyr'])
        
        # storage of the fixational point
        peak_x = int(salmax_coords[1])
        peak_y = int(salmax_coords[0])
        attention_xy.append((peak_x, peak_y))

        # Apply a color map to the window for better visualization
        window_map_jet = cv2.applyColorMap(window.detach().cpu().numpy().squeeze(0).astype(np.uint8), cv2.COLORMAP_JET)
        raw_frame = window_map_jet.copy()
        
        # Add a title to the visualization
        cv2.putText(window_map_jet, 'Events map', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

        # Draw a circle at the location of maximum saliency on the visualization
        cv2.circle(window_map_jet, (salmax_coords[1], salmax_coords[0]), 6, (255, 255, 255), 4)
        cv2.putText(window_map_jet, 'Visual Attention', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Display the events map and saliency map
        cv2.imshow('Events map and Saliency Map', window_map_jet)

        # Wait for a key press to update the display
        cv2.waitKey(1)
        
        # Crops
        x_in = max(peak_x - half_x, 0)
        x_fn = min(peak_x + half_x, max_x)
        y_in = max(peak_y - half_y, 0)
        y_fn = min(peak_y + half_y, max_y)
        crops.append(raw_frame[y_in:y_fn, x_in:x_fn])

        # Increment the time by the window period for the next iteration
        time += window_period

        # Reset the window for the next time period
        window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

# Clean up by closing the OpenCV window after processing all events
cv2.destroyAllWindows()

# %% FIXATIONS AND SACCADES

# Convert history to numpy array
attention_xy = np.array(attention_xy)

# ========== Plot 1: saccadic path ==========
fig1, ax1 = plt.subplots(figsize=(6, 6))

# line + scatter
ax1.plot(attention_xy[:, 0], attention_xy[:, 1],
         '-', alpha=0.5, linewidth=2, color='royalblue')
ax1.scatter(attention_xy[:, 0], attention_xy[:, 1],
            s=80, edgecolors='white', linewidth=1, color='royalblue')

ax1.set_aspect('equal', 'box')
ax1.set_xlim(0, max_x-1)
ax1.set_ylim(max_y-1, 0)    # invert y-axis
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.tick_params(direction='in', length=6, width=2,
                labelbottom=False, labelleft=False)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.show()

# %% Regions of interest (ROIs)

roi_indices = [0, 2, 38, 67]
roi_labels = [rf'$\mathrm{{ROI}}_{{{i + 1}}}$' for i in roi_indices]
selected_crops = [crops[i] for i in roi_indices]

n = len(selected_crops)
fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

for ax, crop_bgr, label in zip(axes, selected_crops, roi_labels):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)  # convertir BGR → RGB
    ax.imshow(crop_rgb)
    ax.set_title(label, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% SAVE

# Create output folder
out_dir = os.path.join("objectData", object_id)
os.makedirs(out_dir, exist_ok=True)

# Save attentional fixational points as .pkl
with open(os.path.join(out_dir, "fixational_xy.pkl"), "wb") as f:
    pkl.dump(attention_xy, f)
    
# Save attentional ROIs as .pkl
with open(os.path.join(out_dir, "ROIs.pkl"), "wb") as f:
    pkl.dump(crops, f)  