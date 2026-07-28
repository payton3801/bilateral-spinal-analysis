# %%
# # ignore this for now
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# import pandas as pd
# import numpy as np
# import os
# import pickle as pkl

# session_id = '053'    
# side = 'right'         
# channel_id = '0'   

# all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'


# if os.path.exists(all_ds_path):
#      with open(all_ds_path, "rb") as f:
#         all_ds = pkl.load(f)

# # Get specific session
# if session_id not in all_ds:
#     raise ValueError(f"Session {session_id} not found in dataset")

# dataset = all_ds[session_id]

# # ==========================================
# # 3. CONFIGURE SIDE (LEFT vs RIGHT)
# # ==========================================
# if side == 'left':
#     dataset.trial_info = dataset.l_trial_info
#     signal_type = 'lfads_rates_L'
# elif side == 'right':
#     dataset.trial_info = dataset.r_trial_info
#     signal_type = 'lfads_rates_R'
# else:
#     raise ValueError("Side must be 'left' or 'right'")

# # Ensure start/end times exist for duration calculation
# dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
# dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']

# trial_data = dataset.make_trial_data(
#     align_field='ext_start_time',
#     align_range=(-200, 800) 
# )

# rates = trial_data[signal_type][channel_id].values
# times = trial_data['align_time'].dt.total_seconds().values

# durations = (dataset.trial_info['end_time'] - dataset.trial_info['start_time']).dt.total_seconds().values


# # Setup Colormap
# cmap = plt.get_cmap('viridis')
# norm = mcolors.Normalize(vmin=np.min(durations), vmax=np.max(durations))

# fig, ax = plt.subplots(figsize=(12, 7))

# # Identify trial boundaries (where time resets)
# split_indices = np.where(np.diff(times) < 0)[0] + 1
# boundary_indices = list(split_indices) + [len(times)]

# start_idx = 0
# for trial_idx, end_idx in enumerate(boundary_indices):
#     # Slice data for this specific trial
#     t_chunk = times[start_idx:end_idx]
#     y_chunk = rates[start_idx:end_idx]
    
#     # Determine color based on step duration
#     if trial_idx < len(durations):
#         step_len = durations[trial_idx]
#         line_color = cmap(norm(step_len))
#     else:
#         line_color = 'gray' # Fallback if sizes mismatch
    
#     # Plot line
#     ax.plot(t_chunk, y_chunk, color=line_color, linewidth=1, alpha=0.6)
#     start_idx = end_idx

# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([]) 
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Step Duration in seconds', rotation=270, labelpad=15)

# # Compute and plot the average firing rate across all trials
# # Compute the average firing rate across all trials
# trial_rates = []  # List to store rates for each trial
# start_idx = 0

# for end_idx in boundary_indices:
#     # Slice the rates for this trial
#     trial_rates.append(rates[start_idx:end_idx])
#     start_idx = end_idx

# # Pad trials to the same length (if necessary) and compute the mean
# from itertools import zip_longest
# padded_trials = np.array(list(zip_longest(*trial_rates, fillvalue=np.nan)))  # Pad with NaN
# avg_rates = np.nanmean(padded_trials, axis=1)  # Compute mean, ignoring NaN

# # Plot the average firing rate
# ax.plot(times[:len(avg_rates)], avg_rates, color='red', linewidth=2)
# ax.legend()

# # Labels
# ax.axvline(0, color='black', linestyle='--', label='Extension Onset')
# ax.set_title(f"LFADS Rates (Session {session_id}")
# ax.set_xlabel("Time from Extension")
# ax.set_ylabel("Firing Rate ")

# plt.tight_layout()
# plt.show()













# %%
# smoothed emg psths

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import pickle as pkl


session_id = '013'
side = 'left'

# --- DIFFERENCE #1: MUSCLE NAME (String) ---
channel_id = ('model_emg_smooth_30ms', 'LVL')

all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

if 'all_ds' not in locals():
    if os.path.exists(all_ds_path):
        print(f"Loading all_ds from {all_ds_path}...")
        with open(all_ds_path, "rb") as f:
            all_ds = pkl.load(f)

dataset = all_ds[session_id]

if side.lower() == 'left':
    dataset.trial_info = dataset.l_trial_info
elif side.lower() == 'right':
    dataset.trial_info = dataset.r_trial_info
else:
    raise ValueError("Side must be 'left' or 'right'")


dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']    

trial_data = dataset.make_trial_data(
    align_field='start_time',
    align_range=(-200, 200) 
)
raw_data = trial_data[channel_id].values
rates = np.abs(raw_data) 

times = trial_data['align_time'].dt.total_seconds().values
durations = (dataset.trial_info['end_time'] - dataset.trial_info['start_time']).dt.total_seconds().values


cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=np.min(durations), vmax=np.max(durations))

fig, ax = plt.subplots(figsize=(12, 7))

split_indices = np.where(np.diff(times) < 0)[0] + 1
boundary_indices = list(split_indices) + [len(times)]

start_idx = 0
for trial_idx, end_idx in enumerate(boundary_indices):
    t_chunk = times[start_idx:end_idx]
    y_chunk = rates[start_idx:end_idx]
    
    if trial_idx < len(durations):
        step_len = durations[trial_idx]
        line_color = cmap(norm(step_len))
    else:
        line_color = 'gray'
    
    ax.plot(t_chunk, y_chunk, color=line_color, linewidth=1, alpha=0.6)
    start_idx = end_idx


# We calculate a simple binned average to show the trend on top of the spaghetti
common_time = np.linspace(-0.2, 0.2, 100) # Match your align_range
mean_trace = []

for t in common_time:
    # Find points close to t (tolerance = 2ms)
    indices = np.where(np.abs(times - t) < 0.002)[0]
    if len(indices) > 0:
        mean_trace.append(np.mean(rates[indices]))
    else:
        mean_trace.append(np.nan)

ax.plot(common_time, mean_trace, color='red', linewidth=3, label='Average Envelope')

# Format
ax.axvline(0, color='black', linestyle='--', label='Stance Onset')
ax.set_title(f"EMG: {channel_id} , Session {session_id}")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()






# %%
# denoised emg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import pickle as pkl
from itertools import zip_longest

file_path = "/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/merged_cat03_013.pkl"
with open(file_path, "rb") as f:
    dataset = pkl.load(f)

muscle_name = 'LVL' 

session_id = '013'    
side = 'left'  # 'left' will plot indices 0-6, 'right' will plot 7-13

muscle_map = {
    'left': {'LVL': '6', 'LTA': '5', 'LMG': '2'}, 
    'right': {'RVL': '13', 'RTA': '12', 'RMG': '9'}
}
channel_id = muscle_map[side.lower()][muscle_name]

# Use the name generated by your merge script
signal_type = 'deEMG_mean_EMG' 
all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

with open(all_ds_path, "rb") as f:
    all_ds = pkl.load(f)

dataset = all_ds[session_id]

# Set trial info based on side
if side.lower() == 'left':
    dataset.trial_info = dataset.l_trial_info
else:
    dataset.trial_info = dataset.r_trial_info

dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']

trial_data = dataset.make_trial_data(
    align_field='start_time',
    align_range=(-200, 200) 
)

# Access the MultiIndex: (SignalName, ChannelIndex)
# Use a tuple to access the MultiIndex levels (Level 0, Level 1)
target_col = (signal_type, channel_id)
rates = trial_data[target_col].values
times = trial_data['align_time'].dt.total_seconds().values

durations = np.abs((dataset.trial_info['end_time'] - dataset.trial_info['start_time']).dt.total_seconds().values) #taking abs value

fig, ax = plt.subplots(figsize=(12, 7))
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=np.min(durations), vmax=np.max(durations))

split_indices = np.where(np.diff(times) < 0)[0] + 1
boundary_indices = list(split_indices) + [len(times)]

trial_rates = []
start_idx = 0
for trial_idx, end_idx in enumerate(boundary_indices):
    t_chunk = times[start_idx:end_idx]
    y_chunk = rates[start_idx:end_idx]
    trial_rates.append(y_chunk)
    
    if trial_idx < len(durations):
        line_color = cmap(norm(durations[trial_idx]))
    else:
        line_color = 'gray'
    
    ax.plot(t_chunk, y_chunk, color=line_color, linewidth=1, alpha=0.4)
    start_idx = end_idx

padded_trials = np.array(list(zip_longest(*trial_rates, fillvalue=np.nan)))
avg_rates = np.nanmean(padded_trials, axis=1)
unique_times = times[:len(avg_rates)] 

ax.plot(unique_times, avg_rates, color='red', linewidth=3, label='Mean Denoised EMG', zorder=5)

# Formatting
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, ax=ax, label='Step Duration (s)')
ax.axvline(0, color='black', label='Stance Onset')
ax.set_title(f"EMG: {signal_type} , Session {session_id}")
ax.set_xlabel("Time from Stance Onset (s)")
ax.set_ylabel("Amplitude (A.U.)")
ax.legend()

plt.tight_layout()
plt.show()



# %%
# sisde by side analysis
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import pickle as pkl
from itertools import zip_longest

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
session_id = '013'    
side = 'left' 
muscle_name = 'LVL' 
all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

# Mapping for the denoised (deEMG) channel indices
muscle_map_denoised = {
    'left': {'LVL': '6', 'LTA': '5', 'LMG': '2'}, 
    'right': {'RVL': '13', 'RTA': '12', 'RMG': '9'}
}

# Load Dataset
if os.path.exists(all_ds_path):
    with open(all_ds_path, "rb") as f:
        all_ds = pkl.load(f)
else:
    raise FileNotFoundError(f"Could not find {all_ds_path}")

dataset = all_ds[session_id]

# Set side-specific trial info
if side.lower() == 'left':
    dataset.trial_info = dataset.l_trial_info
else:
    dataset.trial_info = dataset.r_trial_info

# Fix timestamps for trial alignment
dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']
durations = np.abs((dataset.trial_info['end_time'] - dataset.trial_info['start_time']).dt.total_seconds().values)

# Setup 1x2 Figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=np.min(durations), vmax=np.max(durations))

# Define the two signal keys
# Panel 0: Smoothed EMG, Panel 1: Denoised (LFADS) EMG
signals_to_plot = [
    ('model_emg_smooth_30ms', muscle_name),
    ('deEMG_mean_EMG', muscle_map_denoised[side.lower()][muscle_name])
]
titles = [f"Smoothed EMG", f"Denoised EMG"]

# ==========================================
# 2. PROCESSING & PLOTTING LOOP
# ==========================================
for i, (sig_type, chan_id) in enumerate(signals_to_plot):
    ax = axes[i]
    
    # Generate trial data (ensuring clean alignment)
    trial_data = dataset.make_trial_data(align_field='start_time', align_range=(-200, 200))
    
    # Extract aligned rates and times from the same subset
    target_col = (sig_type, chan_id)
    rates = trial_data[target_col].values
    times = trial_data['align_time'].dt.total_seconds().values
    
    # Identify trial boundaries
    split_indices = np.where(np.diff(times) < 0)[0] + 1
    boundary_indices = [0] + list(split_indices) + [len(times)]
    
    trial_rates_list = []
    for t_idx in range(len(boundary_indices) - 1):
        s, e = boundary_indices[t_idx], boundary_indices[t_idx+1]
        t_chunk, y_chunk = times[s:e], rates[s:e]
        trial_rates_list.append(y_chunk)
        
        # Plot individual trial lines
        line_color = cmap(norm(durations[t_idx])) if t_idx < len(durations) else 'gray'
        ax.plot(t_chunk, y_chunk, color=line_color, linewidth=0.8, alpha=0.3)

    # Compute and plot the Mean (Red Line)
    padded = np.array(list(zip_longest(*trial_rates_list, fillvalue=np.nan)))
    avg = np.nanmean(padded, axis=1)
    
    # Generate clean time vector to prevent looping
    unique_times = np.linspace(-0.2, 0.2, len(avg))
    ax.plot(unique_times, avg, color='red', linewidth=3, label='Mean Trace', zorder=10)

    # Aesthetics
    ax.axvline(0, color='black', linestyle='--', label='Stance Onset')
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("Time from Stance (s)")
    if i == 0:
        ax.set_ylabel("Amplitude (A.U.)")
    ax.legend(loc='upper right')

# Final Formatting
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, label='Step Duration (s)', aspect=30, pad=0.02)

plt.suptitle(f"EMG Smoothed vs Denoised, Session {session_id}, Muscle {muscle_name}", fontsize=16)
plt.show()















# %%
# psths across all sessions
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import pickle as pkl
from itertools import zip_longest

# ==========================================
# 1. CONFIGURATION
# ==========================================
side = 'left'
all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

ALIGN_RANGE = (-200, 800)
START_S, END_S = ALIGN_RANGE[0] / 1000.0, ALIGN_RANGE[1] / 1000.0
STANCE_DURATION_THRESHOLD = 100
muscle_map_denoised = {
    'left': {
        'LVL': '6',
        'LTA': '5',
        'LMG': '2',
        'LBA': '0',
        'LBP': '1',
        'LSA': '3',
        'LSL': '4'
    },
    'right': {
        'RVL': '13',
        'RTA': '12',
        'RMG': '9',
        'RBA': '7',
        'RBP': '8',
        'RSA': '10',
        'RSL': '11'
    }
}

# Load dataset
if os.path.exists(all_ds_path):
    with open(all_ds_path, "rb") as f:
        all_ds = pkl.load(f)
else:
    raise FileNotFoundError(f"Could not find {all_ds_path}")

# ==========================================
# 2. DATA AGGREGATION
# # ==========================================
keys = ['smooth', 'denoised']
all_data = {k: {m: {'rates': [], 'durations': []} for m in muscle_map_denoised[side].keys()} for k in keys}

for session_id, dataset in all_ds.items():
    print(f"Processing session: {session_id}")
    dataset.trial_info = dataset.l_trial_info
    dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
    dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']

    durations = np.abs((dataset.trial_info['end_time'] - dataset.trial_info['start_time']).dt.total_seconds().values)
    trial_data = dataset.make_trial_data(align_field='start_time', align_range=ALIGN_RANGE)

    session_times = trial_data['align_time'].dt.total_seconds().values
    split_indices = np.where(np.diff(session_times) < 0)[0] + 1
    boundaries = [0] + list(split_indices) + [len(session_times)]

    for muscle_name, chan_idx in muscle_map_denoised[side].items():
        smooth_col = ('model_emg_smooth_30ms', muscle_name)
        denoised_col = ('deEMG_mean_EMG', chan_idx)

        if smooth_col in trial_data.columns and denoised_col in trial_data.columns:
            s_sig = np.abs(trial_data[smooth_col].values)
            d_sig = trial_data[denoised_col].values

            # Interpolate denoised to match smoothed resolution
            if len(d_sig) != len(s_sig):
                f = interp1d(np.linspace(0, 1, len(d_sig)), d_sig, kind='linear', fill_value="extrapolate")
                d_sig = f(np.linspace(0, 1, len(s_sig)))

            for t_idx in range(len(boundaries) - 1):
                dur = durations[t_idx]
                if dur <= STANCE_DURATION_THRESHOLD:
                    s, e = boundaries[t_idx], boundaries[t_idx+1]
                    t_trial = session_times[s:e]

                    all_data['smooth'][muscle_name]['rates'].append((t_trial, s_sig[s:e]))
                    all_data['denoised'][muscle_name]['rates'].append((t_trial, d_sig[s:e]))
                    all_data['smooth'][muscle_name]['durations'].append(dur)
                    all_data['denoised'][muscle_name]['durations'].append(dur)


# ==========================================
# 3. PLOTTING AVERAGE SIGNALS
# ==========================================
n_muscles = len(muscle_map_denoised[side])
fig, axes = plt.subplots(2, n_muscles, figsize=(32, 12), sharex=True, sharey='row')
cmap, norm = plt.get_cmap('viridis'), mcolors.Normalize(vmin=0.2, vmax=0.6)
titles = ["Smoothed", "Denoised"]

for col_idx, muscle_name in enumerate(muscle_map_denoised[side].keys()):
    for row_idx, key in enumerate(keys):
        ax = axes[row_idx, col_idx]
        rates = all_data[key][muscle_name]['rates']
        durs = all_data[key][muscle_name]['durations']

        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(START_S, END_S)
        if row_idx == 0: ax.set_title(muscle_name, fontsize=16, fontweight='bold')
        if col_idx == 0: ax.set_ylabel(f"{titles[row_idx]}\nAmplitude (A.U.)")

        if not rates: continue

        # Plot Individual Trials
        for i, (t, y) in enumerate(rates):
            ax.plot(t, y, color=cmap(norm(durs[i])), alpha=0.1, linewidth=0.5)

        # Plot Population Mean
        y_vals = [r[1] for r in rates]
        padded = np.array(list(zip_longest(*y_vals, fillvalue=np.nan)))
        ax.plot(rates[0][0][:len(padded)], np.nanmean(padded, axis=1), color='red', linewidth=2.5)

plt.suptitle(f"{side} Side: Smoothed vs.  Denoised", fontsize=22, y=0.98)
plt.show()










# # %%   COMPLETE STABILIZED WATERFALL PLOT
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import colorcet as cc
# import numpy as np
# import pickle as pkl
# import os

# # --- 1. DATA LOADING ---
# session_id = '061'
# channel_id = ('model_emg_smooth_30ms', 'LVL')
# all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'
# side = 'left' 

# if 'all_ds' not in locals():
#     if os.path.exists(all_ds_path):
#         print(f"Loading all_ds from {all_ds_path}...")
#         with open(all_ds_path, "rb") as f:
#             all_ds = pkl.load(f)

# dataset = all_ds[session_id]
# dataset.trial_info = dataset.l_trial_info if side.lower() == 'left' else dataset.r_trial_info

# # --- 2. ALIGNMENT & TRIAL EXTRACTION ---
# dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
# dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']

# # Ensure time ordering
# dataset.trial_info['start_time'], dataset.trial_info['end_time'] = (
#     np.minimum(dataset.trial_info['start_time'], dataset.trial_info['end_time']),
#     np.maximum(dataset.trial_info['start_time'], dataset.trial_info['end_time']),
# )


# # %%   OVERLAID MULTI-MUSCLE WATERFALL (COMPRESSED VERSION)
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import colorcet as cc
# import numpy as np
# import pickle as pkl
# import os

# # --- 1. DATA LOADING ---
# session_id = '061'
# left_muscles = ['LMG', 'LSL', 'LVL', 'LBP', 'LTA', 'LBA', 'LSA']
# all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

# if 'all_ds' not in locals():
#     if os.path.exists(all_ds_path):
#         with open(all_ds_path, "rb") as f:
#             all_ds = pkl.load(f)

# dataset = all_ds[session_id]
# dataset.trial_info = dataset.l_trial_info # Focusing on the left side

# # --- 2. DYNAMIC ALIGNMENT & EXTRACTION ---
# trial_data = dataset.make_trial_data(
#     start_field='ext_start_time', 
#     end_field='ext_stop_time',
#     margin=100, # 100ms margin for "flattening" at ends
#     allow_overlap=True  
# )

# align_times = trial_data['align_time'].dt.total_seconds().values
# durations = (dataset.trial_info['ext_stop_time'] - dataset.trial_info['ext_start_time']).dt.total_seconds().values

# # --- 3. GLOBAL AGGREGATION & CREATING e_ix ---
# all_left_trials = []

# for muscle in left_muscles:
#     channel_id = ('model_emg_smooth_30ms', muscle)
#     raw_rates = np.abs(trial_data[channel_id].values)
    
#     # Boundary detection for spinal neuron activity segments
#     split_indices = np.where(np.diff(align_times) < 0)[0] + 1
#     boundary_indices = [0] + list(split_indices) + [len(align_times)]
#     rates_split = [(align_times[s:e], raw_rates[s:e]) for s, e in zip(boundary_indices[:-1], boundary_indices[1:])]

#     for i, (rate_tuple, dur) in enumerate(zip(rates_split, durations)):
#         # Filtering to the specific pulsatile range (205-785ms)
#         # Change this line to include your full duration range
#         if 0 <= dur <= 5: 
#             t_vec, dat = rate_tuple
            
#             # CREATING THE INTEGER: This corresponds to the sample index of the burst end
#             e_ix = np.argmin(np.abs(t_vec - dur)) 
            
#             all_left_trials.append({
#                 't_vec': t_vec * 1000, # Convert to ms for plotting
#                 'dat': dat,
#                 'duration': dur * 1000, # Duration in ms
#                 'e_ix': e_ix,
#                 'muscle': muscle
#             })

# # GLOBAL SORT: Ensures the S-curve trajectory follows duration
# all_left_trials.sort(key=lambda x: x['duration'])

# # --- 4. THE PLOTTING LOOP (MATCHING FIGURE 2A) ---
# fig, ax = plt.subplots(figsize=(2.25, 3.25), dpi=300)

# # Constants from your Source Data reference
# spacing = 0.2
# scaling = 0.05
# lw = 0.5
# ms = 1
# cm = cc.m_bmy # Colorcet colormap

# for i, trial in enumerate(all_left_trials):
#     t_vec = trial['t_vec']
#     dat = trial['dat']
#     e_ix = trial['e_ix']
    
#     # Unified laddering offset from Fig2A logic
#     v_offset = (i * spacing) * scaling
    
#     # 1. Trace: Low alpha black line
#     ax.plot(t_vec, dat + v_offset, linewidth=lw, color="k", alpha=0.1)
    
#     # 2. Red Dot: Plotted using the created e_ix integer
#     ax.plot(t_vec[e_ix], dat[e_ix] + v_offset, 'o', color="r", alpha=0.6, markersize=ms)
    
#     # 3. Start Marker: Blue square at alignment point (t=0)
#     start_ix = np.argmin(np.abs(t_vec))
#     ax.plot(t_vec[start_ix], dat[start_ix] + v_offset, 's', color='blue', alpha=0.6, markersize=ms+2)

# # --- 5. FORMATTING & STYLE ---
# # Green line for alignment (t=0)
# ax.axvline(x=0, color='g', linestyle='--', linewidth=0.8, alpha=0.5)

# ax.set_xlim(-150, 3000)
# ax.axis('off')

# # Custom Scale Bar
# ax.plot([-150, -50], [-0.05, -0.05], linewidth=1.5, color="k", clip_on=False)
# ax.text(-100, -0.1, "100ms", ha="center", va="top", fontsize=8)

# plt.tight_layout()
# plt.show()


























# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as colormaps # Renamed to avoid confusion with the function cm
import pandas as pd
import numpy as np
import os
import pickle as pkl
import colorcet as cc # Required for cc.m_bmy or high-quality colormaps

all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

if os.path.exists(all_ds_path):
    with open(all_ds_path, "rb") as f:
        all_ds = pkl.load(f)
    print("all_ds loaded successfully.")
else:
    raise FileNotFoundError(f"File not found at {all_ds_path}")

# Selecting the specific dataset
ds = all_ds['037']

def sort_and_reorder_steps(trial_info):
    # Ensure we are working on a copy to avoid SettingWithCopy warnings
    df = trial_info.copy()
    combined_times = pd.concat([
        df['ext_start_time'], 
        df['ext_stop_time']
    ]).sort_values().reset_index(drop=True)
    
    df['ext_start_time'] = combined_times.iloc[::2].reset_index(drop=True)
    df['ext_stop_time'] = combined_times.iloc[1::2].reset_index(drop=True)
    return df

# Apply the sorting function
l_trial_info = sort_and_reorder_steps(l_trial_info)
r_trial_info = sort_and_reorder_steps(r_trial_info)

# Update the dataset object (using 'ds', not 'dataset')
ds.l_trial_info = l_trial_info
ds.r_trial_info = r_trial_info

# --- 2. DATA PROCESSING ---
# Calculate burst duration
ds.r_trial_info['burst_duration'] = (ds.r_trial_info['ext_stop_time'] - ds.r_trial_info['ext_start_time'])
ds.l_trial_info['burst_duration'] = (ds.l_trial_info['ext_stop_time'] - ds.l_trial_info['ext_start_time'])


# Sort trial info by duration for the waterfall plot
sorted_info = ds.r_trial_info.sort_values('burst_duration').copy()

# Re-calculate indices based on sorted order
# Ensure ds.data.index is sorted for 'nearest' to work reliably
onsets = ds.data.index.get_indexer(sorted_info['ext_start_time'], method='nearest')
offsets = ds.data.index.get_indexer(sorted_info['ext_stop_time'], method='nearest')

# Ensure durations are floats (seconds) for normalization
if isinstance(sorted_info['burst_duration'].iloc[0], pd.Timedelta):
    durations_float = sorted_info['burst_duration'].dt.total_seconds().values
else:
    durations_float = sorted_info['burst_duration'].values

print("Processing complete. Ready for plotting.")

# --- 3. PLOT CONFIGURATION ---
spacing = 1.0  # Increased for better vertical separation
scaling = 0.08 # Adjusted for EMG amplitude
lw = 0.5 
ms = 1.5 
tick_fs = 8
# Use colorcet or standard viridis
cm = colormaps.get_cmap('viridis') 

pre_ms = 150 
post_ms = 3000

# Calculate bin_ms from the data index
bin_ms = (ds.data.index[1] - ds.data.index[0]).total_seconds() * 1000

# --- 4. SETUP FIGURE ---
fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
ax.axis('off')
n_cycles = len(sorted_info)

# --- 5. THE WATERFALL LOOP ---
for i in range(n_cycles):
    on_idx = onsets[i]
    off_idx = offsets[i]
    
    start_win = on_idx - int(pre_ms / bin_ms)
    end_win = on_idx + int(post_ms / bin_ms)
    
    # Safety Check: Skip if trial is too close to boundaries
    if start_win < 0 or end_win > len(ds.data):
        continue

    sig = ds.data['emg']['RSL'].values[start_win:end_win]
    if len(sig) == 0:
        continue
        
    v_offset = (i * spacing) * scaling
    t_vec = np.linspace(-pre_ms, post_ms, len(sig))
    
    # A. Plot the Trace
    ax.plot(t_vec, sig + v_offset, linewidth=lw, color="k", alpha=0.2)
    
    # B. Plot the Red Dot (Offset/End of burst)
    rel_off_idx = off_idx - start_win
    if 0 <= rel_off_idx < len(sig):
        ax.plot(t_vec[rel_off_idx], sig[rel_off_idx] + v_offset, 'o', 
                color="r", alpha=0.7, markersize=ms)
    
    # C. Plot the Colored Square at Alignment Point (t=0)
    zero_idx = int(pre_ms / bin_ms)
    if zero_idx < len(sig):
        denom = durations_float.max() - durations_float.min()
        norm_dur = (durations_float[i] - durations_float.min()) / (denom if denom > 0 else 1.0)
        ax.plot(0, sig[zero_idx] + v_offset, 's', 
                color=cm(norm_dur), alpha=0.8, markersize=ms+1)

# --- 6. ANNOTATIONS ---
ax.axvline(x=0, color="g", linestyle="--", alpha=0.5, linewidth=1)

# 100ms Scale Bar
y_base = -0.1
ax.plot([-150, -50], [y_base, y_base], color="k", linewidth=2, clip_on=False)
ax.text(-100, y_base - 0.2, "100ms", ha="center", va="top", fontsize=tick_fs)

# Duration labels for start/end
ax.text(-160, 0, f"{int(durations_float.min()*1000)}ms", ha='right', fontsize=tick_fs, va='center')
ax.text(-160, (n_cycles * spacing) * scaling, f"{int(durations_float.max()*1000)}ms", ha='right', fontsize=tick_fs, va='center')
ax.set_xlim(-200, 2000)

plt.tight_layout()
plt.show()






















# %%
# single cycle activation plots 

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import pandas as pd
# import numpy as np
# import os
# import pickle as pkl
# import colorcet as cc
# from scipy.ndimage import gaussian_filter1d

# # --- 1. DATA LOADING ---
# all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

# if os.path.exists(all_ds_path):
#     with open(all_ds_path, "rb") as f:
#         all_ds = pkl.load(f)
#     print("all_ds loaded successfully.")
# else:
#     raise FileNotFoundError(f"File not found at {all_ds_path}")

# # --- 2. DATA AGGREGATION & CLEANING ---
# all_trials_list = []

# for session_id, ds_obj in all_ds.items():

    
#     r_info = ds_obj.r_trial_info.copy()
#     r_info['session_id'] = session_id
#     #l_info = ds_obj.r_trial_info.copy()
#     #l_info['session_id'] = session_id
    
#     # Calculate duration
#     durations = (r_info['ext_stop_time'] - r_info['ext_start_time'])
#     if isinstance(durations.iloc[0], pd.Timedelta):
#         r_info['burst_duration'] = durations.dt.total_seconds()
#     else:
#         r_info['burst_duration'] = durations
        
#     all_trials_list.append(r_info)
        
#     # durations = (l_info['ext_stop_time'] - l_info['ext_start_time'])
#     # if isinstance(durations.iloc[0], pd.Timedelta):
#     #     l_info['burst_duration'] = durations.dt.total_seconds()
#     # else:
#     #     l_info['burst_duration'] = durations
        
#     # all_trials_list.append(l_info)

# # Combine all sessions
# all_trials = pd.concat(all_trials_list, ignore_index=True)

# initial_count = len(all_trials)
# all_trials = all_trials[all_trials['burst_duration'] > 0]
# # print(f"Removed {initial_count - len(all_trials)} invalid trials with negative durations.")

# all_trials = pd.concat(all_trials_list, ignore_index=True)

# # 2. SET BOUNDS: Calculate 5th and 95th percentiles (The "Lahiru Fix")
# # This removes the top and bottom 5% of "weird" trials
# # lower_bound = all_trials['burst_duration'].quantile(0.00)
# # upper_bound = all_trials['burst_duration'].quantile(0.90)

# # Alternatively, if you know the specific range for your feline data:
# lower_bound = 0.300  # 150ms
# upper_bound = 1.150  

# all_trials = all_trials[(all_trials['burst_duration'] >= lower_bound) & 
#                         (all_trials['burst_duration'] <= upper_bound)]

# print(f"Filtering trials between {int(lower_bound*1000)}ms and {int(upper_bound*1000)}ms")

# # Sort by duration for the waterfall effect
# sorted_info = all_trials.sort_values('burst_duration').reset_index(drop=True)
# durations_float = sorted_info['burst_duration'].values

# # Sort by duration for the waterfall effect
# sorted_info = all_trials.sort_values('burst_duration').reset_index(drop=True)
# durations_float = sorted_info['burst_duration'].values

# # --- 3. PLOT CONFIGURATION (DENSE LOOK) ---
# spacing = 0.01   # Small spacing for overlapping "mountain" look
# scaling = 0.02   # Height of the raw EMG spikes
# lw = 0.5        # Thin lines for high density
# alpha_val = 0.07 # Transparency to allow overlaps to show depth
# ms = 0.8         # Small dots for burst offsets
# tick_fs = 8

# pre_ms = 200 
# post_ms = 3500  # Increased to ensure long bursts aren't cut off

# try:
#     cm = cc.cm['m_bmy']
# except KeyError:
#     cm = cc.cm['bmy']

# # --- 4. SETUP FIGURE ---
# fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
# ax.axis('off')
# n_cycles = len(sorted_info)
# norm = plt.Normalize(vmin=durations_float.min(), vmax=durations_float.max())

# # --- 5. THE WATERFALL LOOP ---
# for i, (idx, row) in enumerate(sorted_info.iterrows()):
#     sess_id = row['session_id']
#     ds_obj = all_ds[sess_id]
    
#     # Get Sampling Rate
#     dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
#     on_idx = ds_obj.data.index.get_indexer([row['ext_start_time']], method='nearest')[0]
#     off_idx = ds_obj.data.index.get_indexer([row['ext_stop_time']], method='nearest')[0]
    
#     start_win = on_idx - int(pre_ms / dt_ms)
#     end_win = on_idx + int(post_ms / dt_ms)
    
#     if start_win < 0 or end_win > len(ds_obj.data):
#         continue

#     # Pull RAW EMG (No smoothing)
#     sig = ds_obj.data['emg']['RSL'].values[start_win:end_win]
#     #sig = ds_obj.data['emg']['LSL'].values[start_win:end_win]

#     if len(sig) == 0: continue
    
#     v_offset = i * spacing
#     t_vec = np.linspace(-pre_ms, post_ms, len(sig))

#     # A. Plot Raw Trace
#     ax.plot(t_vec, (sig * scaling) + v_offset, 
#             linewidth=lw, color="black", alpha=alpha_val, zorder=1)
    
#     # B. Plot Red Dot at Burst Offset (Now strictly to the right of green line)
#     rel_off_time = row['burst_duration'] * 1000
#     stop_idx_in_win = np.argmin(np.abs(t_vec - rel_off_time))
    
#     if 0 <= stop_idx_in_win < len(sig):
#         ax.plot(t_vec[stop_idx_in_win], (sig[stop_idx_in_win] * scaling) + v_offset, 
#                 'o', color="red", markersize=ms, alpha=0.6, zorder=5)
    
#     # C. Plot Continuous Vertical Color Bar
#     color = cm(norm(row['burst_duration']))
#     ax.plot([0, 0], [v_offset, v_offset + spacing], 
#             color=color, linewidth=5, solid_capstyle='butt', zorder=6)

# # --- 6. ANNOTATIONS ---
# ax.axvline(x=0, color="green", linestyle="--", alpha=0.4, linewidth=1, zorder=0)

# # Scale Bar
# y_bar = -spacing * 15
# ax.plot([0, 100], [y_bar, y_bar], color="black", linewidth=2.5)
# ax.text(50, y_bar - (spacing * 10), "100ms", ha="center", va="top", fontsize=tick_fs, fontweight='bold')

# # Duration Labels
# ax.text(-80, 0, f"{int(durations_float.min()*1000)}ms", ha='right', va='center', fontsize=tick_fs)
# ax.text(-80, (n_cycles-1) * spacing, f"{int(durations_float.max()*1000)}ms", ha='right', va='center', fontsize=tick_fs)

# # Final formatting
# ax.set_xlim(-pre_ms - 50, post_ms +100)
# plt.tight_layout()
# plt.show()






















# %%
# single cycle activation plots with cutoff between steps

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle as pkl
import colorcet as cc

# --- 1. DATA LOADING ---
all_ds_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets/all_ds.pkl'

if os.path.exists(all_ds_path):
    with open(all_ds_path, "rb") as f:
        all_ds = pkl.load(f)
    print("all_ds loaded successfully.")
else:
    raise FileNotFoundError(f"File not found at {all_ds_path}")

# --- 2. DATA AGGREGATION & CLEANING (LSL Example) ---
all_trials_list = []

for session_id, ds_obj in all_ds.items():

    
    # Using LSL (Left Side) - Change to RSL/RTA as needed
    info = ds_obj.l_trial_info.copy()
    info['session_id'] = session_id
    
    # Calculate duration and convert to float
    durations = (info['ext_stop_time'] - info['ext_start_time'])
    if isinstance(durations.iloc[0], pd.Timedelta):
        info['burst_duration'] = durations.dt.total_seconds()
    else:
        info['burst_duration'] = durations
        
    all_trials_list.append(info)

# Combine and Filter
# all_trials = pd.concat(all_trials_list, ignore_index=True)
# # Sanity Filter: Remove negative durations and extreme outliers (> 6.5s)
# all_trials = all_trials[(all_trials['burst_duration'] > 0) & (all_trials['burst_duration'] < 6.5)]

# # Sort by duration for the waterfall effect
# sorted_info = all_trials.sort_values('burst_duration').reset_index(drop=True)
# durations_float = sorted_info['burst_duration'].values

# --- 2. DATA AGGREGATION & CLEANING ---
all_trials = pd.concat(all_trials_list, ignore_index=True)

# 2. SET BOUNDS: Calculate 5th and 95th percentiles (The "Lahiru Fix")
# This removes the top and bottom 5% of "weird" trials
# lower_bound = all_trials['burst_duration'].quantile(0.00)
# upper_bound = all_trials['burst_duration'].quantile(0.90)

# Alternatively, if you know the specific range for your feline data:
lower_bound = 0.300  # 150ms
upper_bound = 1.150  

all_trials = all_trials[(all_trials['burst_duration'] >= lower_bound) & 
                        (all_trials['burst_duration'] <= upper_bound)]

print(f"Filtering trials between {int(lower_bound*1000)}ms and {int(upper_bound*1000)}ms")

# Sort by duration for the waterfall effect
sorted_info = all_trials.sort_values('burst_duration').reset_index(drop=True)
durations_float = sorted_info['burst_duration'].values


# --- 3. PLOT CONFIGURATION ---
spacing = 0.003   # Small spacing for overlapping "mountain" look
scaling = 0.02   # Height of the raw EMG spikes
lw = 0.5        # Thin lines for high density
alpha_val = 0.07 # Transparency to allow overlaps to show depth
ms = 0.8         # Small dots for burst offsets
tick_fs = 8

pre_ms = 200 
post_burst_buffer = 150 # How many ms to show AFTER the red dot
post_ms_max = int(durations_float.max() * 1000) + post_burst_buffer 

try:
    cm = cc.cm['m_bmy']
except KeyError:
    cm = cc.cm['bmy']

# --- 4. SETUP FIGURE ---
fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
ax.axis('off')

alpha_val = 0.7 

n_cycles = len(sorted_info)
norm = plt.Normalize(vmin=durations_float.min(), vmax=durations_float.max())

# --- 5. THE DYNAMIC WATERFALL LOOP ---
for i, (idx, row) in enumerate(sorted_info.iterrows()):
    sess_id = row['session_id']
    ds_obj = all_ds[sess_id]
    
    dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
    on_idx = ds_obj.data.index.get_indexer([row['ext_start_time']], method='nearest')[0]
    
    # Extract the full window for safety
    start_win = on_idx - int(pre_ms / dt_ms)
    end_win = on_idx + int(post_ms_max / dt_ms)
    
    if start_win < 0 or end_win > len(ds_obj.data):
        continue

    sig = ds_obj.data['emg']['LSL'].values[start_win:end_win]
    t_vec = np.linspace(-pre_ms, post_ms_max, len(sig))
    v_offset = i * spacing

    # --- DYNAMIC MASKING ---
    # Calculates the end point for THIS specific trial trace
    burst_end_ms = row['burst_duration'] * 1000
    plot_cutoff_ms = burst_end_ms + post_burst_buffer
    mask = t_vec <= plot_cutoff_ms
    
    # A. Plot Masked Raw Trace
    ax.plot(t_vec[mask], (sig[mask] * scaling) + v_offset, 
            linewidth=lw, color="black", alpha=alpha_val, zorder=1)
    
    # B. Plot Red Dot at Burst Offset
    stop_idx = np.argmin(np.abs(t_vec - burst_end_ms))
    if 0 <= stop_idx < len(sig):
        ax.plot(t_vec[stop_idx], (sig[stop_idx] * scaling) + v_offset, 
                'o', color="red", markersize=ms *7, alpha=0.7, zorder=5)
    
    # C. Plot Continuous Vertical Color Bar
    color = cm(norm(row['burst_duration']))
    ax.plot([0, 0], [v_offset, v_offset + spacing], 
            color=color, linewidth=5, solid_capstyle='butt', zorder=6)

# --- 6. ANNOTATIONS ---
# Add a vertical green dashed line at x=0
ax.axvline(x=0, color="green", linestyle="--", alpha=0.4, linewidth=1, zorder=0)

# Scale Bar
y_bar = -spacing * 25
ax.plot([0, 100], [y_bar, y_bar], color="black", linewidth=2.5, alpha=alpha_val)  # Apply alpha here
ax.text(50, y_bar - (spacing * 10), "100ms", ha="center", va="top", fontsize=tick_fs, fontweight='bold')

# Duration Labels
ax.text(-80, 0, f"{int(durations_float.min()*1000)}ms", ha='right', va='center', fontsize=tick_fs, alpha=alpha_val)  # Apply alpha here
ax.text(-80, (n_cycles-1) * spacing, f"{int(durations_float.max()*1000)}ms", ha='right', va='center', fontsize=tick_fs, alpha=alpha_val)  # Apply alpha here


# Final formatting: Ensure window is tight around data
ax.set_xlim(-pre_ms - 50, post_ms_max)
plt.tight_layout()
plt.show()



# %%
####labels after removing outliers

# List of indices to exclude
exclude_indices = [2, 4, 22, 49, 61, 62, 68, 70, 81]

# Filter out the unwanted indices from sorted_info
filtered_info = sorted_info.drop(exclude_indices).reset_index(drop=True)

# --- 1. SETUP FIGURE ---
fig, ax = plt.subplots(figsize=(10, 20), dpi=300) 
ax.axis('off')

# Get global plot limits for normalization
durations_float = filtered_info['burst_duration'].values
norm = plt.Normalize(vmin=durations_float.min(), vmax=durations_float.max())

# --- 2. THE DYNAMIC WATERFALL LOOP (LABEL ALL DOTS) ---
for i, (idx, row) in enumerate(filtered_info.iterrows()):
    sess_id = str(row['session_id'])
    ds_obj = all_ds[row['session_id']]
    
    dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
    on_idx = ds_obj.data.index.get_indexer([row['ext_start_time']], method='nearest')[0]
    
    # Calculate window
    t_vec = np.linspace(-pre_ms, post_ms_max, int((pre_ms + post_ms_max)/dt_ms))
    start_win = on_idx - int(pre_ms / dt_ms)
    end_win = start_win + len(t_vec)
    
    if start_win < 0 or end_win > len(ds_obj.data): 
        continue

    sig = ds_obj.data['emg']['LSL'].values[start_win:end_win]
    v_offset = i * spacing
    burst_end_ms = row['burst_duration'] * 1000

    # A. Plot Raw EMG Trace
    mask = t_vec <= (burst_end_ms + post_burst_buffer)
    ax.plot(t_vec[mask], (sig[mask] * scaling) + v_offset, 
            linewidth=lw, color="black", alpha=alpha_val, zorder=1)
    
    # B. Plot Red Dot
    stop_idx = np.argmin(np.abs(t_vec - burst_end_ms))
    dot_x, dot_y = t_vec[stop_idx], (sig[stop_idx] * scaling) + v_offset
    ax.plot(dot_x, dot_y, 'o', color="red", markersize=ms*10, alpha=0.9, zorder=5)

    # --- C. LABEL EVERY DOT ---
    # Using a tiny font and offset to the right
    # label_str = f"#{i}: {int(burst_end_ms)}ms (S:{sess_id[-3:]})"
    # ax.text(dot_x + 10, dot_y, label_str, 
    #         fontsize=4, color='darkblue', va='center', ha='left',
    #         bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1),
    #         zorder=10)

    # D. Color Bar
    color = cm(norm(row['burst_duration']))
    ax.plot([0, 0], [v_offset, v_offset + spacing], color=color, linewidth=6, zorder=6)

# --- 3. FINAL FORMATTING ---
ax.axvline(x=0, color="green", linestyle="--", alpha=0.3, linewidth=1)
ax.set_xlim(-pre_ms - 20, post_ms_max + 400) # Buffer for the labels
plt.tight_layout()
plt.show()









# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

# --- 1. CONFIGURATION ---
align_field = 'ext_start_time'  # Alignment field
pre_ms = 200  # Time before alignment (ms)
post_ms = 800  # Time after alignment (ms)
scaling = 0.02  # Scale EMG amplitude for better visualization
sigma = 5  # Standard deviation for Gaussian smoothing

# --- 2. DATA AGGREGATION ---
aligned_emg = []  # Store aligned EMG signals
burst_durations = []  # Store burst durations
time_vector = None  # Time vector for plotting

for i, (idx, row) in enumerate(sorted_info.iterrows()):
    sess_id = row['session_id']
    ds_obj = all_ds[sess_id]
    
    # Get sampling rate (ms per bin)
    dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
    
    # Get alignment index
    on_idx = ds_obj.data.index.get_indexer([row[align_field]], method='nearest')[0]
    
    # Define window around alignment
    start_win = on_idx - int(pre_ms / dt_ms)
    end_win = on_idx + int(post_ms / dt_ms)
    
    if start_win < 0 or end_win > len(ds_obj.data):
        continue  # Skip trials that exceed data boundaries
    
    # Extract EMG signal (e.g., 'LSL' muscle)
    emg_signal = ds_obj.data['deEMG_var_EMG']['4'].values[start_win:end_win]

    
    # Normalize time vector
    t_vec = np.linspace(-pre_ms, post_ms, len(emg_signal))
    
    # Store aligned EMG signal and burst duration
    aligned_emg.append(emg_signal)
    burst_durations.append(row['burst_duration'])
    if time_vector is None:
        time_vector = t_vec

# --- 3. SMOOTH AND PLOT ---
aligned_emg = np.array(aligned_emg)  # Convert to NumPy array
burst_durations = np.array(burst_durations)

# Normalize burst durations for colormap
norm = mcolors.Normalize(vmin=burst_durations.min() * 1000, vmax=burst_durations.max() * 1000)
cmap = cm.get_cmap('plasma')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each aligned and smoothed EMG signal with color based on burst duration
for emg_signal, burst_duration in zip(aligned_emg, burst_durations):
    smoothed_signal = gaussian_filter1d(emg_signal, sigma=sigma)  # Apply Gaussian smoothing
    color = cmap(norm(burst_duration * 1000))  # Map burst duration to colormap
    ax.plot(time_vector, smoothed_signal * scaling, color=color, alpha=0.7, linewidth=1)

# Add color bar for burst durations
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
cbar.set_label('Extensor Burst Duration (ms)')

# Formatting
ax.axvline(0, color='black', label='Extension Onset')
ax.set_title('Smoothed and Aligned EMG PSTH Colored by Burst Duration (LSL)', fontsize=14)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('EMG Amplitude ', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# plotted by averages ~5 lines
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

# --- 1. CONFIGURATION ---
align_field = 'ext_start_time'   # Alignment point (0 ms on plot)
offset_field = 'ext_stop_time'    # Biological termination point
pre_ms = 250                     # Time shown before onset
buffer_ms = 100                  # Short buffer after offset
scaling = 0.02                   # Adjust for visual height of EMG
num_bins = 6                     # Number of duration groups for colormap

# --- 2. DATA AGGREGATION (Variable Length Slicing) ---
aligned_emg = []  
burst_durations = []  
time_vectors = [] 

for i, (idx, row) in enumerate(sorted_info.iterrows()):
    sess_id = row['session_id']
    ds_obj = all_ds[sess_id]
    
    # Calculate sampling interval (ms per bin)
    dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
    
    # 2a. Calculate Start Index (Onset minus pre_ms)
    on_idx = ds_obj.data.index.get_indexer([row[align_field]], method='nearest')[0]
    start_win = on_idx - int(pre_ms / dt_ms)
    
    # 2b. Calculate End Index (Offset plus short buffer)
    off_idx = ds_obj.data.index.get_indexer([row[offset_field]], method='nearest')[0]
    end_win = off_idx + int(buffer_ms / dt_ms)
    
    # Boundary check to ensure we stay within the recording session
    if start_win < 0 or end_win > len(ds_obj.data):
        continue 
    
    # 2c. Extract deEMG_var_EMG signal (component 7)
    emg_signal = ds_obj.data['deEMG_var_EMG']['4'].values[start_win:end_win]
    
    # Generate a unique time vector for this trial's specific length
    t_vec = np.arange(-pre_ms, (len(emg_signal) * dt_ms) - pre_ms, dt_ms)
    t_vec = t_vec[:len(emg_signal)]  # Exact length match for plotting
    
    aligned_emg.append(emg_signal)
    burst_durations.append(row['burst_duration'])
    time_vectors.append(t_vec)

# --- 3. BINNING AND AVERAGING ---
burst_durations = np.array(burst_durations)
# Define bins based on the distribution of burst durations
bins = np.linspace(burst_durations.min(), burst_durations.max(), num_bins + 1)
bin_indices = np.digitize(burst_durations, bins)

# --- 4. PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# Normalize colormap for the legend (converting to ms)
norm = mcolors.Normalize(vmin=burst_durations.min() * 1000, vmax=burst_durations.max() * 1000)
cmap = cm.get_cmap('plasma')

for b in range(1, len(bins)):
    # Mask for trials in this specific duration bin
    bin_mask = (bin_indices == b)
    if not np.any(bin_mask):
        continue
    
    # Count the number of trials in this bin
    bin_count = np.sum(bin_mask)
    print(f"Bin {b}: {bin_count} trials")
    
    # Extract trials and durations
    bin_signals = [aligned_emg[j] for j, val in enumerate(bin_mask) if val]
    bin_durs = burst_durations[bin_mask]
    
    # 4a. Handle variable lengths for the bin-average line
    # We use the median length of trials in this bin to represent the average trace
    target_len = int(np.median([len(s) for s in bin_signals]))
    
    processed_signals = []
    for s in bin_signals:
        if len(s) >= target_len:
            processed_signals.append(s[:target_len])
        else:
            # Pad with the last recorded value to keep the mean stable
            processed_signals.append(np.pad(s, (0, target_len - len(s)), 'edge'))
            
    mean_sig = np.mean(processed_signals, axis=0)
    sem_sig = np.std(processed_signals, axis=0) / np.sqrt(len(processed_signals))  # Calculate SEM
    mean_time = np.arange(-pre_ms, (target_len * dt_ms) - pre_ms, dt_ms)[:target_len]
    
    # 4b. Plot the raw mean signal for the bin (no smoothing)
    avg_dur_ms = np.mean(bin_durs) * 1000
    color = cmap(norm(avg_dur_ms))
    
    # Plot the mean signal
    ax.plot(mean_time, mean_sig * scaling, 
            color=color, linewidth=2.5, alpha=0.9, label=f'{avg_dur_ms:.0f} ms')
    
    # Plot the SEM as a lighter shaded region
    ax.fill_between(mean_time, (mean_sig - sem_sig) * scaling, (mean_sig + sem_sig) * scaling, 
                    color=color, alpha=0.1)  # Lighter shading

# --- 5. FINAL FORMATTING ---
# Horizontal Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.18, aspect=40)
cbar.set_label('Average Burst Duration (ms)', fontsize=11)

# Annotations
ax.axvline(0, color='black', linestyle='--', alpha=0.6, lw=1.5)
ax.text(5, ax.get_ylim()[1]*0.85, '', fontweight='bold', alpha=0.7)

ax.set_title('deEMG_var_EMG LSL PSTH', fontsize=14, pad=15)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude (a.u.)', fontsize=12)

# Set x-limit to accommodate the longest burst + buffer
ax.set_xlim(-pre_ms, (burst_durations.max() * 1000) + buffer_ms + 100)

# Clean "paper-ready" style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()



# %%
## neural psths
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

# --- 1. CONFIGURATION ---
# We pick '051' because it has 140 trials (the most data)
target_session = '051' 

align_field = 'ext_start_time'   
offset_field = 'ext_stop_time'    
pre_ms = 250                     
buffer_ms = 100                  
num_bins = 6                     

# Define the units you want in your 6 panels
target_units = ['0', '1', '2', '3', '4', '5'] 

# --- 2. DATA AGGREGATION (SINGLE SESSION ONLY) ---
# FIX: Filter sorted_info to ONLY the target session before starting
session_info = sorted_info[sorted_info['session_id'] == target_session].copy()

aligned_data = []  
burst_durations = []  

for i, (idx, row) in enumerate(session_info.iterrows()):
    # All trials in this loop now belong to target_session
    ds_obj = all_ds[target_session]
    
    # Calculate sampling interval
    dt_ms = (ds_obj.data.index[1] - ds_obj.data.index[0]).total_seconds() * 1000
    
    # Calculate Window Indices
    on_idx = ds_obj.data.index.get_indexer([row[align_field]], method='nearest')[0]
    start_win = on_idx - int(pre_ms / dt_ms)
    off_idx = ds_obj.data.index.get_indexer([row[offset_field]], method='nearest')[0]
    end_win = off_idx + int(buffer_ms / dt_ms)
    
    if start_win < 0 or end_win > len(ds_obj.data):
        continue 

    # Extract the neural rates for the units we want
    # Since we are in one session, we can trust these columns exist
    trial_rates = ds_obj.data['lfads_rates_L'][target_units].iloc[start_win:end_win]
    
    aligned_data.append(trial_rates)
    burst_durations.append(row['burst_duration'])

# --- 3. BINNING (Specific to Session 051) ---
burst_durations = np.array(burst_durations)
# Define bins based only on the range of durations seen in this session
bins = np.linspace(burst_durations.min(), burst_durations.max(), num_bins + 1)
bin_indices = np.digitize(burst_durations, bins)

# --- 4. PLOTTING (The 6-Panel Figure) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, dpi=120)
axes = axes.flatten()

norm = mcolors.Normalize(vmin=burst_durations.min() * 1000, vmax=burst_durations.max() * 1000)
cmap = cm.get_cmap('plasma')

for i, unit_id in enumerate(target_units):
    ax = axes[i]
    
    for b in range(1, len(bins)):
        # Mask for trials in this bin
        bin_mask = (bin_indices == b)
        
        # Advice Check: Skip if this duration group doesn't exist for this session
        if not np.any(bin_mask):
            continue
            
        bin_signals = [aligned_data[j][unit_id].values for j, is_in_bin in enumerate(bin_mask) if is_in_bin]
        
        # Handle variable lengths
        target_len = int(np.median([len(s) for s in bin_signals]))
        processed_signals = []
        for s in bin_signals:
            if len(s) >= target_len:
                processed_signals.append(s[:target_len])
            else:
                processed_signals.append(np.pad(s, (0, target_len - len(s)), 'edge'))
                
        mean_sig = np.mean(processed_signals, axis=0)
        sem_sig = np.std(processed_signals, axis=0) / np.sqrt(len(processed_signals))
        mean_time = np.arange(-pre_ms, (target_len * dt_ms) - pre_ms, dt_ms)[:target_len]
        
        color = cmap(norm(np.mean(burst_durations[bin_mask]) * 1000))
        
        ax.plot(mean_time, mean_sig, color=color, linewidth=2, alpha=0.9)
        ax.fill_between(mean_time, mean_sig - sem_sig, mean_sig + sem_sig, color=color, alpha=0.1)

    # Individual Subplot Formatting
    ax.set_title(f'Unit {unit_id}', fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if i >= 3: ax.set_xlabel('Time (ms)')
    if i % 3 == 0: ax.set_ylabel('Firing Rate')

# --- 5. COLORBAR & FINAL LAYOUT ---
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(f'Average Burst Duration (ms)')

plt.suptitle(f'Neural PSTHs: Session {target_session}', fontsize=16)
plt.subplots_adjust(bottom=0.15, hspace=0.3)
plt.show()
# %%

## pcas
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Iterate through each trial
for trial_idx, trial_pca in enumerate(pca_trials):
    # Get the timestamps for this specific trial to determine phase per point
    # Note: You'll need to reference your original ds_obj index here
    
    # Create segments for the line
    points = trial_pca.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # --- LOGIC TO COLOR PER POINT ---
    # You need a color array the same length as the trial
    # Example: red for the first half (flexion), blue for second (extension)
    colors = []
    mid_point = len(trial_pca) // 2 
    for i in range(len(trial_pca) - 1):
        if i < mid_point:
            colors.append((1, 0, 0, 0.3)) # Red (RGBA)
        else:
            colors.append((0, 0.4, 1, 0.3)) # Blue (RGBA)
    
    # Create the 3D collection
    lc = Line3DCollection(segments, colors=colors, linewidth=1.5)
    ax.add_collection3d(lc)

# Auto-scale the axes since collections don't do it automatically
ax.set_xlim(pca_data[:,0].min(), pca_data[:,0].max())
ax.set_ylim(pca_data[:,1].min(), pca_data[:,1].max())
ax.set_zlim(pca_data[:,2].min(), pca_data[:,2].max())

ax.set_title('Interneuron State Space Trajectories (Phase-Colored)')
plt.show()

# %%



from sklearn.decomposition import PCA

target_session = '051'
target_units = ['0', '1', '2', '3', '4', '5']

all_neural = np.vstack([d[target_units].values for d in aligned_data])
pca = PCA(n_components=3)
pca.fit(all_neural)

pca_trials = [pca.transform(d[target_units].values) for d in aligned_data]
pca_data = np.vstack(pca_trials)

print(f"pca_trials built: {len(pca_trials)} trials")
print(f"Variance explained: {pca.explained_variance_ratio_}")
## pcas
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Remove outlier trials ---
bad_trials = [6, 13]
pca_trials_clean = [t for i, t in enumerate(pca_trials) if i not in bad_trials]
burst_durations_clean = np.array([d for i, d in enumerate(burst_durations) if i not in bad_trials])
pca_data_clean = np.vstack(pca_trials_clean)

zero_idx = int(pre_ms / dt_ms)  # index corresponding to t=0 (extension onset)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for trial_idx, (trial_pca, dur) in enumerate(zip(pca_trials_clean, burst_durations_clean)):
    points = trial_pca.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = []
    for i in range(len(trial_pca) - 1):
        if i < zero_idx:
            colors.append((0.85, 0.1, 0.1, 0.4))   # red: pre-extension
        else:
            colors.append((0.1, 0.3, 0.9, 0.4))    # blue: post-extension

    lc = Line3DCollection(segments, colors=colors, linewidth=1.2)
    ax.add_collection3d(lc)

# --- Equal aspect ratio ---
max_range = np.array([
    pca_data_clean[:, 0].max() - pca_data_clean[:, 0].min(),
    pca_data_clean[:, 1].max() - pca_data_clean[:, 1].min(),
    pca_data_clean[:, 2].max() - pca_data_clean[:, 2].min()
]).max() / 2.0

mid = [pca_data_clean[:, i].mean() for i in range(3)]
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Interneuron State Space Trajectories\n(Red = Pre-Extension, Blue = Post-Extension)')

plt.tight_layout()
plt.show()

print(f"Plotting {len(pca_trials_clean)} trials (removed {len(bad_trials)} outliers)")
print(f"Duration range: {burst_durations_clean.min()*1000:.0f}ms – {burst_durations_clean.max()*1000:.0f}ms")
# %%
