# %%
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- CONFIG ---
# We now know these are the correct pairs
truth_key = 'model_emg_smooth_30ms'
pred_key = 'deEMG_mean_ALL' 
channel_idx = 3  # Channel 0 looked great, so let's stick with it
t_slice = slice(800, 1000) # Zoom in on that burst we saw (frame 800-1000)

# --- EXTRACT ---
# Get the raw data
actual_emg = dataset.data[truth_key].iloc[:, channel_idx]
pred_emg = dataset.data[pred_key].iloc[:, channel_idx]

# --- PLOT ---
plt.figure(figsize=(10, 5))
plt.plot(actual_emg[t_slice].values, color='black', alpha=0.3, label='Actual EMG (Smoothed)')
plt.plot(pred_emg[t_slice].values, color='red', linewidth=2.5, label='LFADS Prediction')

plt.title(f"EMG Decoding Performance: Channel {channel_idx}")
plt.xlabel("Time Bins")
plt.ylabel("EMG Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import logging
import sys
from os import path

# Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
file_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/nlb_cat03_013.pkl'

# Comparison Settings
BIN_SIZE_MS = 10         
SIGMA_MS = 50            
sigma_bins = SIGMA_MS / BIN_SIZE_MS

# Fields
RAW_FIELD = 'spikes'
# We will iterate through these to plot both
SIDE_CONFIGS = [
    {'side': 'Left',  'key': 'lfads_rates_L', 'color': 'red'},
    {'side': 'Right', 'key': 'lfads_rates_R', 'color': 'blue'}
]

# Time Slice (Zoom in)
time_slice = slice(1000, 2000) 

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
if not path.exists(file_path):
    logger.error(f"File not found: {file_path}")
    sys.exit()

logger.info(f"Loading dataset: {file_path}")
with open(file_path, "rb") as f:
    dataset = pickle.load(f)

df = dataset.data
logger.info(f"Keys found: {df.columns.levels[0].tolist()}")

# ------------------------------------------------------------------
# 2. PLOT BOTH SIDES
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
conversion_factor = (1000 / BIN_SIZE_MS) # Scale to Hz

for idx, config in enumerate(SIDE_CONFIGS):
    side_name = config['side']
    lfads_key = config['key']
    line_color = config['color']
    ax = axes[idx]

    # Check if this side exists in the dataframe
    if lfads_key not in df.columns.levels[0]:
        ax.text(0.5, 0.5, f"{lfads_key} not found", ha='center')
        logger.warning(f"Skipping {side_name}: Key {lfads_key} missing.")
        continue

    # --- SELECT BEST UNIT FOR THIS SIDE ---
    # Find units that exist in both Raw Spikes and this specific LFADS side
    valid_units = [c for c in df[RAW_FIELD].columns if c in df[lfads_key].columns]
    
    if not valid_units:
        ax.text(0.5, 0.5, "No matching units", ha='center')
        continue

    # Pick the highest firing rate unit on this side
    mean_rates = df[RAW_FIELD][valid_units].mean()
    best_unit = mean_rates.idxmax()
    
    logger.info(f"[{side_name}] Best Unit: {best_unit} | Mean Rate: {mean_rates[best_unit]*conversion_factor:.2f} Hz")

    # --- EXTRACT & PROCESS ---
    raw_trace = df[RAW_FIELD][best_unit].iloc[time_slice].values.astype(float)
    lfads_trace = df[lfads_key][best_unit].iloc[time_slice].values.astype(float)

    # Smooth Raw & Convert to Hz
    raw_smoothed = gaussian_filter1d(raw_trace, sigma=sigma_bins) * conversion_factor
    
    # Convert LFADS to Hz
    lfads_trace_hz = lfads_trace * conversion_factor

    # --- PLOT ---
    ax.plot(raw_smoothed, color='gray', alpha=0.5, label=f'Smoothed {SIGMA_MS}ms)', linewidth=2)
    ax.plot(lfads_trace_hz, color=line_color, alpha=0.9, label=f'LFADS {side_name}', linewidth=2.5)

    # Calc Correlation
    corr = np.corrcoef(raw_smoothed, lfads_trace_hz)[0, 1]

    # Labels
    ax.set_title(f"{side_name} Side: Unit {best_unit}", fontsize=14)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Final Layout
axes[-1].set_xlabel("Time (Bins)", fontsize=12)
plt.tight_layout()
plt.show()
