# %%
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.base import DataWrangler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import glob
import os
import _pickle as pickle
import logging
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as colormap
import h5py
from sklearn.linear_model import Ridge


# %%
# --- setup logger -- these give more info than print statements
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% --- load dataset
og_ds_name = 'cat03_039'
base_path = "/snel/share/share/derived/auyong/NWB/" 
cache_base = "/snel/share/share/tmp/scratch/pbechef/" 
nwb_path = f"/snel/share/share/derived/auyong/NWB/{og_ds_name}.nwb"
#nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/r_{og_ds_name}_nwb/"
nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/" 
dataset = NWBDataset(nwb_path)
BIN_SIZE = dataset.bin_width 
align_file_suffix = "_low_reg"

names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
           'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
           'cat03_013', 'cat03_025', 'cat03_049']

# %% --- apply filtering
#cutoff_freq = 5 # Hz
env_emg_gauss_width_ms = 200  # ms
gauss_width_ms = 100  # ms
emg_name = 'emg'
spk_name = 'spikes'
smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
emg_field = smth_emg_field
smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"

#saving this preprocessed data
if not os.path.exists(nwb_cache_dir):
    os.makedirs(nwb_cache_dir)

save_filename = f"{og_ds_name}_preprocessed.pkl"
save_path = os.path.join(nwb_cache_dir, save_filename)

#check if preprocessed dataset exists
if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        dataset = pickle.load(f)
    logger.info(f"Loaded preprocessed dataset from {save_path}")
else:
    dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

    #saving dataset
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f"Saved preprocessed dataset to {save_path}")

# %% --- [COMPUTE] compute onsets/offset events

def compute_on_off_events(ds, musc_name_right , pos_threshold=0.025, neg_threshold=0.03):
    # get data
    dat = ds.data[emg_field][musc_name_right ] #raw signal data
    env = ds.data[envl_emg_field][musc_name_right ] #enveloped signal data
    print(f"Data shape: {dat.shape}, Envelope shape: {env.shape}")

    if musc_name_right not in ds.data[emg_field].columns:
        raise KeyError(f"Column '{musc_name_right}' not found in {emg_field}")

    def diff_filter(x): # computes first derivative
        """differentation filter"""
        return signal.savgol_filter(x, 7, 5, deriv=1, axis=0)

    def flip(x):
        return -1 * x

    # compute diff of muscle activation trace
    diff = dat.to_frame().apply(diff_filter) # creates dataframe, creates diff variable to make first derivative
    #pos_threshold = 0.025
    #neg_threshold = 0.03
    min_dist_ms = 60  # min ms between change pts
    min_dist = np.round(min_dist_ms / BIN_SIZE).astype(int) #converts to number of samples
    # use find peaks to identify positive peaks in diff
    pos_peaks = diff.apply(signal.find_peaks, height=pos_threshold).iloc[0][0]    
    neg_peaks = diff.apply(flip).apply(signal.find_peaks, height=neg_threshold).iloc[0][0]

    # use find peaks to find troughs in envelope
    change_points = (
        env.apply(flip)
        .to_frame()
        .apply(signal.find_peaks, distance=min_dist, prominence=np.nanvar(env) * 1.5).iloc[ #indices of troughs in the envelope signal
            0
        ][0]
    )
    
    onsets = []
    offsets = []
    # between two change points 
    # onset: find the first positive peak that occurs after first change pt
    # offset: find the last negative peak that occurs before last change pt
    for i in range(change_points.size - 1):
        # -- onset calculation
        p_ix = np.where(pos_peaks > change_points[i])[0] #pos peaks occuring after change point
        onset_cand = pos_peaks[p_ix[0]]
        if onset_cand < change_points[i + 1]: # checks if occurs before next change point
            onset = onset_cand
        else:
            onset = np.nan
        # -- offset calculation
        n_ix = np.where(neg_peaks < change_points[i + 1])[0] #neg peaks occur after change point
        offset_cand = neg_peaks[n_ix[-1]]
        if offset_cand > change_points[i]: # checks if occurs before next change point
            offset = offset_cand + 3
        else:
            offset = np.nan
        # -- check that onset and offset were calculated
        test_nan = [onset, offset] #onset and offset lists are created if value returned isnt nan
        if np.all(~np.isnan(test_nan)):
            onsets.append(onset)
            offsets.append(offset)
    # create a "debug package" that stores any additional information that 
    # isn't necessarily needed for the function's purpose, but could be 
    # data or information that could be helpful for diagnostics on the function
    # that could be useful for modifying parameters
    debug_pkg = dict()
    debug_pkg["data"] = dat
    debug_pkg["envelope"] = env
    debug_pkg["diff"] = diff.squeeze()
    debug_pkg["pos_peaks"] = pos_peaks
    debug_pkg["neg_peaks"] = neg_peaks
    debug_pkg["change_points"] = change_points 

    print(f"Onsets: {onsets}, Offsets: {offsets}, Debug Package: {debug_pkg}")

    return np.array(onsets), np.array(offsets), debug_pkg

# %%

# TODO: rename output r_ext_on, r_ext_off, r_ext_db_pkg #SL/ BA / MG / SA
musc_name_right = 'RSL'
r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)

# TODO: rename output l_ext_on, l_ext_off, l_ext_db_pkg
musc_name_left = 'LSL'
l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)


# %% --- DEBUG: compute onsets/offset for extensor muscle (find appropriate threshold for data)
#all the code and plots used to make aligned plots
#RIGHT SIDE

# choose signal to perform computation
emg_field = smth_emg_field

# choose muscle name
musc_name_right = "RSL"
musc_name_left = "LSL"

# extract signal from dataframe
sig = dataset.data[smth_emg_field][musc_name_right]
env_sig = dataset.data[envl_emg_field][musc_name_right]


## creates differentiated and smoothed emg signal
def diff_filter(x):
    """differentation filter"""
    return signal.savgol_filter(x, 7, 5, deriv=1, axis=0)

## ---------------------------------------------------------------------------- ##
# differentiating and amplifying the signal in order to find change points. 
# they are used as boundary points on our original signal, so that each onset and offset are only registered once for each spike
d_sig = sig.to_frame().apply(diff_filter)
fig = plt.figure(figsize=(10,3), dpi=150)
ax = fig.add_subplot(111)
ax.plot(sig) # sig: raw signal
ax.plot(env_sig*-1) # smoothed signal, simplified representation of the signal's overall shape 
ax.plot(d_sig) # differentiated signal, showing rate of change, inflection points
ax.set_ylim([-0.005, 0.005])
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
# raw signal plotted over enveloped signal, local minimas marked with red dot
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
ax.plot(r_ext_db_pkg['data'].values, color='k')
ax.plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(r_ext_db_pkg['change_points'],r_ext_db_pkg['envelope'][r_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
#  plotting differentiated amplified signal on top of the last plot
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
ax.plot(r_ext_db_pkg['data'].values, color='k')
ax.plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(r_ext_db_pkg['diff'].values*100, color='b', alpha=0.3) # diff signal amplified
ax.plot(r_ext_db_pkg['change_points'],r_ext_db_pkg['envelope'][r_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
# plotting 
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
amp_diff = r_ext_db_pkg['diff'].values*100
ax.plot(r_ext_db_pkg['data'].values, color='k', zorder=3)
ax.plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(amp_diff, color='b', alpha=0.3)
ax.plot(r_ext_db_pkg['change_points'],r_ext_db_pkg['envelope'][r_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.plot(r_ext_db_pkg['pos_peaks'], amp_diff[r_ext_db_pkg['pos_peaks']], 'o', color='c', markersize=2, alpha=0.3) # plotting all pos peaks on diff signal
ax.plot(r_ext_db_pkg['neg_peaks'], amp_diff[r_ext_db_pkg['neg_peaks']], 'o', color='m', markersize=2, alpha=0.3) # plotting all neg peaks on diff signal
ax.plot(r_ext_on, r_ext_db_pkg['data'][r_ext_on], 'o', color='y', markersize=5, zorder=4)
ax.plot(r_ext_off, r_ext_db_pkg['data'][r_ext_off], 'o', color='y', markersize=5, zorder=4)
#ax.plot(ext_db_pkg['neg_peaks'], amp_diff[ext_db_pkg['neg_peaks']], 'o', color='c', markersize=5)
## ---------------------------------------------------------------------------- ##


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
amp_diff = r_ext_db_pkg['diff'].values*100
ax.plot(r_ext_db_pkg['data'].values, color='k', zorder=3)
ax.plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(amp_diff, color='b', alpha=0.3)
ax.plot(r_ext_db_pkg['change_points'],r_ext_db_pkg['envelope'][r_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.plot(r_ext_db_pkg['pos_peaks'], amp_diff[r_ext_db_pkg['pos_peaks']], 'o', color='c', markersize=2, alpha=0.3)
ax.plot(r_ext_db_pkg['neg_peaks'], amp_diff[r_ext_db_pkg['neg_peaks']], 'o', color='m', markersize=2, alpha=0.3)
ax.plot(r_ext_on, r_ext_db_pkg['data'][r_ext_on], 'o', color='g', markersize=5, zorder=4)
ax.plot(r_ext_off, r_ext_db_pkg['data'][r_ext_off], 'o', color='r', markersize=5, zorder=4)
#ax.plot(ext_db_pkg['neg_peaks'], amp_diff[ext_db_pkg['neg_peaks']], 'o', color='c', markersize=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure(figsize=(4,3), dpi=150)
ax = fig.add_subplot(111)
pre_idx = 400
post_idx = 700

for idx_on in r_ext_on:    
    ax.plot(r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx])

test = r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(test)

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(test - 0.4)

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(np.sign(test-0.4))

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(np.diff(np.sign(test-0.4)))


#%%

env_emg_gauss_width_ms = 200  # ms
gauss_width_ms = 100  # ms
emg_name = 'emg'
spk_name = 'spikes'
smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
emg_field = smth_emg_field
smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"

for name in names:
    # Load the dataset for the current name
    nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
    dataset = NWBDataset(nwb_path)
    BIN_SIZE = dataset.bin_width

    # Apply smoothing functions
    dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

    # Compute onsets/offsets for the right muscle
    musc_name_right = 'RSL'
    musc_name_left = 'LSL'

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=150)

    # Plot data on the first subplot
    amp_diff = r_ext_db_pkg['diff'].values * 100
    axes[0].plot(r_ext_db_pkg['data'].values, color='k', zorder=3)
    axes[0].plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
    axes[0].plot(r_ext_on, r_ext_db_pkg['data'][r_ext_on], 'o', color='g', markersize=5, zorder=4)
    axes[0].plot(r_ext_off, r_ext_db_pkg['data'][r_ext_off], 'o', color='r', markersize=5, zorder=4)

    # Customize subplot appearance
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[0].set_title(f"Raw Signal R side for {name}")

    l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)

    amp_diff = l_ext_db_pkg['diff'].values * 100
    axes[1].plot(l_ext_db_pkg['data'].values, color='k', zorder=3)
    axes[1].plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
    axes[1].plot(l_ext_on, l_ext_db_pkg['data'][l_ext_on], 'o', color='g', markersize=5, zorder=4)
    axes[1].plot(l_ext_off, l_ext_db_pkg['data'][l_ext_off], 'o', color='r', markersize=5, zorder=4)

    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    axes[1].set_title(f"Raw Signal L side for {name}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    
# %% --- DEBUG: compute onsets/offset for extensor muscle (find appropriate threshold for data)
#all the code and plots used to make aligned plots
#LEFT SIDE

# choose signal to perform computation
emg_field = smth_emg_field

# choose muscle name
#musc_name_left = "LSL"
# extract signal from dataframe
sig = dataset.data[smth_emg_field][musc_name_right]
env_sig = dataset.data[envl_emg_field][musc_name_right]


## creates differentiated and smoothed emg signal
def diff_filter(x):
    """differentation filter"""
    return signal.savgol_filter(x, 7, 5, deriv=1, axis=0)

## ---------------------------------------------------------------------------- ##
# differentiating and amplifying the signal in order to find change points. 
# they are used as boundary points on our original signal, so that each onset and offset are only registered once for each spike
d_sig = sig.to_frame().apply(diff_filter)
fig = plt.figure(figsize=(10,3), dpi=150)
ax = fig.add_subplot(111)
ax.plot(sig) # sig: raw signal
ax.plot(env_sig*-1) # smoothed signal, simplified representation of the signal's overall shape 
ax.plot(d_sig) # differentiated signal, showing rate of change, inflection points
ax.set_ylim([-0.005, 0.005])
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
# raw signal plotted over enveloped signal, local minimas marked with red dot
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
ax.plot(l_ext_db_pkg['data'].values, color='k')
ax.plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(l_ext_db_pkg['change_points'],l_ext_db_pkg['envelope'][l_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
#  plotting differentiated amplified signal on top of the last plot
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
ax.plot(l_ext_db_pkg['data'].values, color='k')
ax.plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(l_ext_db_pkg['diff'].values*100, color='b', alpha=0.3) # diff signal amplified
ax.plot(l_ext_db_pkg['change_points'],l_ext_db_pkg['envelope'][l_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## ---------------------------------------------------------------------------- ##

## ---------------------------------------------------------------------------- ##
# plotting 
fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
amp_diff = l_ext_db_pkg['diff'].values*100
ax.plot(l_ext_db_pkg['data'].values, color='k', zorder=3)
ax.plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(amp_diff, color='b', alpha=0.3)
ax.plot(l_ext_db_pkg['change_points'],l_ext_db_pkg['envelope'][l_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.plot(l_ext_db_pkg['pos_peaks'], amp_diff[l_ext_db_pkg['pos_peaks']], 'o', color='c', markersize=2, alpha=0.3) # plotting all pos peaks on diff signal
ax.plot(l_ext_db_pkg['neg_peaks'], amp_diff[l_ext_db_pkg['neg_peaks']], 'o', color='m', markersize=2, alpha=0.3) # plotting all neg peaks on diff signal
ax.plot(l_ext_on, l_ext_db_pkg['data'][l_ext_on], 'o', color='y', markersize=5, zorder=4)
ax.plot(l_ext_off, l_ext_db_pkg['data'][l_ext_off], 'o', color='y', markersize=5, zorder=4)
#ax.plot(ext_db_pkg['neg_peaks'], amp_diff[ext_db_pkg['neg_peaks']], 'o', color='c', markersize=5)
## ---------------------------------------------------------------------------- ##


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure(figsize=(10, 4), dpi=150)
ax = fig.add_subplot(111)
# signal
amp_diff = l_ext_db_pkg['diff'].values*100
ax.plot(l_ext_db_pkg['data'].values, color='k', zorder=3)
ax.plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
ax.plot(amp_diff, color='b', alpha=0.3)
ax.plot(l_ext_db_pkg['change_points'],l_ext_db_pkg['envelope'][l_ext_db_pkg['change_points']], 'o', color='r', markersize=2, alpha=0.3)
ax.plot(l_ext_db_pkg['pos_peaks'], amp_diff[l_ext_db_pkg['pos_peaks']], 'o', color='c', markersize=2, alpha=0.3)
ax.plot(l_ext_db_pkg['neg_peaks'], amp_diff[l_ext_db_pkg['neg_peaks']], 'o', color='m', markersize=2, alpha=0.3)
ax.plot(l_ext_on, l_ext_db_pkg['data'][l_ext_on], 'o', color='g', markersize=5, zorder=4)
ax.plot(l_ext_off, l_ext_db_pkg['data'][l_ext_off], 'o', color='r', markersize=5, zorder=4)
#ax.plot(ext_db_pkg['neg_peaks'], amp_diff[ext_db_pkg['neg_peaks']], 'o', color='c', markersize=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure(figsize=(4,3), dpi=150)
ax = fig.add_subplot(111)
pre_idx = 400
post_idx = 700

for idx_on in l_ext_on:    
    ax.plot(l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx])

# creates a segment of signal around the onset data
test = l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(test)

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(test - 0.3)

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(np.sign(test-0.3))

fig = plt.figure(figsize=(4,3), dpi=150)
plt.plot(np.diff(np.sign(test-0.3)))

for name in names:
    # Define the path to the preprocessed .pkl file
    preprocessed_path = os.path.join(nwb_cache_dir, f"{name}_preproc.pkl")

    if os.path.exists(preprocessed_path):
        with open(preprocessed_path, 'rb') as f:
            dataset = pickle.load(f)
        logger.info(f"Loaded preprocessed dataset for {name} from {preprocessed_path}")
    else:
        # Handle the case where the preprocessed file does not exist
        logger.warning(f"Preprocessed dataset for {name} not found at {preprocessed_path}. Skipping...")
        continue  # Skip to the next dataset
    BIN_SIZE = dataset.bin_width

    # Apply smoothing functions
    # dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
    # dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
    # dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

    # Compute onsets/offsets for the right and left muscles
    musc_name_right = 'RSL'
    musc_name_left = 'LSL'

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=150)

    # Refine onsets for the right muscle
    pre_idx = 500
    post_idx = 700
    refined_on_r = np.zeros_like(r_ext_on)
    if name == 'cat03_043':
        threshold = 0.3
    elif name in ['cat03_061', 'cat03_025']:
        threshold = 0.1
    else:
        threshold = 0.2
    for i, idx_on in enumerate(r_ext_on):    
        win = r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
        cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
        shift = cross_idx - pre_idx
        refined_on_r[i] = r_ext_on[i] + shift
        ref_on = refined_on_r[i]
        ref_win = r_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
        axes[0].plot(ref_win, alpha=0.7)  # Plot refined signal

    axes[0].set_title(f"R Refined Onsets for {name}")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Refine onsets for the left muscle
    refined_on_l = np.zeros_like(l_ext_on)
    if name in ['cat03_37', 'cat03_043', 'cat03_047', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_013']:
        threshold = 0.3
    else:
        threshold = 0.2
    for i, idx_on in enumerate(l_ext_on):    
        win = l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
        cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
        shift = cross_idx - pre_idx
        refined_on_l[i] = l_ext_on[i] + shift
        ref_on = refined_on_l[i]
        ref_win = l_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
        axes[1].plot(ref_win, alpha=0.7)  # Plot refined signal

    axes[1].set_title(f"L Refined Onsets for {name}")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

#%%

# check if directory exists
if not os.path.isdir(nwb_cache_dir):
    logger.info(f"Creating nwb cache dir at: {nwb_cache_dir}")
    os.makedirs(nwb_cache_dir)
else:
    logger.info(f"nwb cache dir exists at: {nwb_cache_dir}")  

# %%

env_emg_gauss_width_ms = 200  # ms
gauss_width_ms = 100  # ms

emg_name = 'emg'
spk_name = 'spikes'

smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
emg_field = smth_emg_field

smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"

pre_idx=400
post_idx=700

for name in names:
    # nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
    # dataset = NWBDataset(nwb_path)
    # BIN_SIZE = dataset.bin_width

    # # Apply smoothing functions
    # dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
    # dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
    # dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

    # # Compute onsets/offsets for the right and left muscles
    # musc_name_right = 'RSL'
    # musc_name_left = 'LSL'

    # r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)
    # l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)

    def refine_tx(tx, data, threshold, pre_idx, post_idx, tx_type="onset"):
        """refine onset/offset calculation"""
        refined_tx = np.zeros_like(tx)
        for i, idx in enumerate(tx):
            win = data.values[idx-pre_idx:idx+post_idx]
            if tx_type == "onset":
                cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
            elif tx_type == "offset":
                cross_idx = np.where(np.diff(np.sign(win-threshold)) < 0)[0][0]
            else:
                raise NotImplementedError("tx_type must be onset or offset")
            shift = cross_idx - pre_idx
            refined_tx[i] = tx[i] + shift
        
        return refined_tx
    
    def plot_aligned_win(ax, tx, data, pre_idx, post_idx, title=None):
        for t in tx:
            win = data.values[t-pre_idx:t+post_idx]
            ax.plot(win)        
        if title is not None:
            ax.set_title(title)
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # -- refine onsets -- left 
    if name in ['cat03_37', 'cat03_043', 'cat03_047', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_013']:
        threshold = 0.3
    else:
        threshold = 0.2    
    refined_on_l = refine_tx(l_ext_on, l_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="onset")
    # -- refine onsets -- right
    if name == 'cat03_043':
        threshold = 0.3
    elif name in ['cat03_061', 'cat03_025']:
        threshold = 0.1
    else:
        threshold = 0.2
    refined_on_r = refine_tx(r_ext_on, r_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="onset")

    threshold = 0.2
    # -- refine offsets -- left     
    #break
    refined_off_l = refine_tx(l_ext_off, l_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="offset")
    # -- refine offsets -- right 
    refined_off_r = refine_tx(r_ext_off, r_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="offset")

    plot_aligned_win(axes[0,0], refined_on_l, l_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} R Onset")
    plot_aligned_win(axes[0,1], refined_on_r, r_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} L Onset")
    plot_aligned_win(axes[1,0], refined_off_l, l_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} L Offset")
    plot_aligned_win(axes[1,1], refined_off_r, r_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} R Offset")
    
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    
    #make trial_info for left and right sides
    def make_trial_info(ds, refined_on, refined_off):
        # -- make l trial_info
        trial_id = []
        condition_id = []
        on_times = []
        off_times = []
        for i, (on_ix, off_ix) in enumerate(zip(refined_on, refined_off)):
            on_times.append(ds.data.index[on_ix])
            off_times.append(ds.data.index[off_ix])
            trial_id.append(i)
            condition_id.append(1)
        trial_info = pd.DataFrame([trial_id, condition_id, on_times, off_times]).T
        trial_info.columns = ['trial_id', 'condition_id', 'ext_start_time', 'ext_stop_time']
        return trial_info

    l_trial_info = make_trial_info(dataset, refined_on_l, refined_off_l)
    r_trial_info = make_trial_info(dataset, refined_on_r, refined_off_r)
    

    dataset.l_trial_info = l_trial_info
    dataset.r_trial_info = r_trial_info

    save_filename = f"{name}_preproc.pkl"
    pkl_save_path = os.path.join(nwb_cache_dir, save_filename)
    with open(pkl_save_path, 'wb') as f:
        logger.info(f"Saving {save_filename} to pickle.")
        pickle.dump(dataset, f)
    # for onset, offset in zip(refined_on_r, refined_off_r):
    #     burst_durations_r = refined_off_r - refined_on_r
    #     burst_durations_l = refined_off_l - refined_on_l


    # fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=200)
    # ax[0].hist(burst_durations_l, bins=20)
    # ax[0].set_title(f"Burst durations for LSL for {name}")

    # ax[1].hist(burst_durations_r, bins=20)
    # ax[1].set_title(f"Burst durations for RSL for {name}")

    plt.tight_layout()
    plt.show()

#%%
#plotting all burst durations for right and left sides on one plot
datasets = []
all_ds = []
all_names = []

for name in names:
    nwb_path = f"{base_path}{name}.nwb"    
    dataset = NWBDataset(nwb_path)
    datasets.append(dataset)
    all_ds.append(dataset)
    all_names.append(name)

all_burst_durations_left = []
all_burst_durations_right = []

# Iterate through all dataset names
for name in names:
    # check if preprocessed dataset exists
    if os.path.exists(preprocessed_path):
        # Load preprocessed dataset
        with open(preprocessed_path, 'rb') as f:
            dataset = pickle.load(f)
        logger.info(f"Loaded preprocessed dataset for {name}")
    else:
        # if preprocessed dataset doesn't exist
        logger.warning(f"Preprocessed dataset for {name} not found. Recomputing...")
        dataset = NWBDataset(nwb_path)
        BIN_SIZE = dataset.bin_width

        # Apply smoothing functions
        dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
        dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
        dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

        # Compute onsets/offsets for the right and left muscles
        musc_name_right = 'RSL'
        musc_name_left = 'LSL'

        r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)
        l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)

        # Save the preprocessed dataset
        with open(preprocessed_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved preprocessed dataset for {name} to {preprocessed_path}")

    # Refine onsets and offsets for left and right muscles
    pre_idx = 400
    post_idx = 700
    threshold = 0.2

    # Refine left onsets
    refined_on_l = np.zeros_like(l_ext_on)
    for i, idx_on in enumerate(l_ext_on):
        win = l_ext_db_pkg['data'].values[idx_on - pre_idx:idx_on + post_idx]
        cross_idx = np.where(np.diff(np.sign(win - threshold)) > 0)[0][0]
        shift = cross_idx - pre_idx
        refined_on_l[i] = l_ext_on[i] + shift

    # Refine right onsets
    refined_on_r = np.zeros_like(r_ext_on)
    for i, idx_on in enumerate(r_ext_on):
        win = r_ext_db_pkg['data'].values[idx_on - pre_idx:idx_on + post_idx]
        cross_idx = np.where(np.diff(np.sign(win - threshold)) > 0)[0][0]
        shift = cross_idx - pre_idx
        refined_on_r[i] = r_ext_on[i] + shift

    # Refine left offsets
    refined_off_l = np.zeros_like(l_ext_off)
    for i, idx_off in enumerate(l_ext_off):
        win = l_ext_db_pkg['data'].values[idx_off - pre_idx:idx_off + post_idx]
        cross_idx = np.where(np.diff(np.sign(win - threshold)) < 0)[0][0]
        shift = cross_idx - pre_idx
        refined_off_l[i] = l_ext_off[i] + shift

    # Refine right offsets
    refined_off_r = np.zeros_like(r_ext_off)
    for i, idx_off in enumerate(r_ext_off):
        win = r_ext_db_pkg['data'].values[idx_off - pre_idx:idx_off + post_idx]
        cross_idx = np.where(np.diff(np.sign(win - threshold)) < 0)[0][0]
        shift = cross_idx - pre_idx
        refined_off_r[i] = r_ext_off[i] + shift

    # Compute burst durations
    burst_durations_left = refined_off_l - refined_on_l
    burst_durations_right = refined_off_r - refined_on_r

    # Append burst durations to the lists
    all_burst_durations_left.extend(burst_durations_left)
    all_burst_durations_right.extend(burst_durations_right)

# Plot histograms for left and right sides
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

# Left side histogram
axes[0].hist(all_burst_durations_left, bins=40, alpha=0.7)
axes[0].set_title('Burst Durations (Left Side)')
axes[0].set_xlabel('Burst Duration Length')
axes[0].set_ylabel('Frequency')

# Right side histogram
axes[1].hist(all_burst_durations_right, bins=40, alpha=0.7)
axes[1].set_title('Burst Durations (Right Side)')
axes[1].set_xlabel('Burst Duration Length')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% -- update the trial_info table -- left

#using try function to not overwrite trial_info 

try:
    dataset.old_trial_info
    logger.info("old trial info exists, skipping overwrite of trial info")
except AttributeError:
    logger.info('old trial info does not exist, copying over old trial info')
    dataset.old_trial_info = dataset.trial_info # move over old trial info to new attribute
    trial_id = pd.DataFrame(np.arange(refined_on_l.size), columns=['trial_id'])
    cond_id = pd.DataFrame(np.ones_like(trial_id), columns=['condition_id'])
    start_time = dataset.data.reset_index()['clock_time'].iloc[refined_on_l].reset_index().drop(columns=['index']).rename(columns={'clock_time':'start_time'})
    end_time = dataset.data.reset_index()['clock_time'].iloc[refined_off_l].reset_index().drop(columns=['index']).rename(columns={'clock_time':'end_time'})
    dataset.l_trial_info = pd.concat([trial_id, start_time, end_time, cond_id], axis=1)

# %% -- update the trial_info table -- right

#using try function to not overwrite trial_info 
try:
    dataset.old_trial_info
    logger.info("old trial info exists, skipping overwrite of trial info")
except AttributeError:
    logger.info('old trial info does not exist, copying over old trial info')
    dataset.old_trial_info = dataset.trial_info # move over old trial info to new attribute
    trial_id = pd.DataFrame(np.arange(refined_on_r.size), columns=['trial_id'])
    cond_id = pd.DataFrame(np.ones_like(trial_id), columns=['condition_id'])
    start_time = dataset.data.reset_index()['clock_time'].iloc[refined_on_r].reset_index().drop(columns=['index']).rename(columns={'clock_time':'start_time'})
    end_time = dataset.data.reset_index()['clock_time'].iloc[refined_off_r].reset_index().drop(columns=['index']).rename(columns={'clock_time':'end_time'})
    dataset.r_trial_info = pd.concat([trial_id, start_time, end_time, cond_id], axis=1)

    dataset.trial_info = dataset.r_trial_info

# %%
# Iterate through all dataset names
for ds_name in names:  # Use 'ds_name' to represent the current dataset name
    # -- create cache dir if it does not exist
    if not os.path.isdir(nwb_cache_dir):
        logger.info(f"Creating {nwb_cache_dir}")
        os.makedirs(nwb_cache_dir)

    # -- dump dataset object to pickle file
    ds_savepath = os.path.join(nwb_cache_dir, "nwb_" + ds_name + ".pkl")
    
    # Check if the dataset is valid before attempting to save
    if dataset is None or dataset.data.empty:
        logger.error(f"Dataset for {ds_name} is None or empty. Skipping save.")
    else:
        logger.info(f"Saving dataset to {ds_savepath}")

        try:
            # Save the dataset to a pickle file
            with open(ds_savepath, "wb") as rfile:
                pickle.dump(dataset, rfile, protocol=4)
                logger.info(f"Dataset {ds_name} saved to pickle.")
        except Exception as e:
            logger.error(f"Failed to save dataset {ds_name} to {ds_savepath}: {e}")


# %%
#old code


# def compute_on_off_events(ds, m_name, pos_threshold=0.025, neg_threshold=0.03):
#     # get data
#     dat = ds.data[emg_field][m_name] #raw signal data
#     env = ds.data[envl_emg_field][m_name] #enveloped signal data

#     def diff_filter(x): # computes first derivative
#         """differentation filter"""
#         return signal.savgol_filter(x, 7, 5, deriv=1, axis=0)

#     def flip(x):
#         return -1 * x

#     # compute diff of muscle activation trace
#     diff = dat.to_frame().apply(diff_filter) # creates dataframe, creates diff variable to make first derivative
#     #pos_threshold = 0.025
#     #neg_threshold = 0.03
#     min_dist_ms = 60  # min ms between change pts
#     min_dist = np.round(min_dist_ms / BIN_SIZE).astype(int) #converts to number of samples
#     # use find peaks to identify positive peaks in diff
#     pos_peaks = diff.apply(signal.find_peaks, height=pos_threshold).iloc[0][0]
#     # use find peaks to identify negative peaks in diff
#     neg_peaks = diff.apply(flip).apply(signal.find_peaks, height=neg_threshold).iloc[0][0]
#     # use find peaks to find troughs in envelope
#     change_points = (
#         env.apply(flip)
#         .to_frame()
#         .apply(signal.find_peaks, distance=min_dist, prominence=np.nanvar(env) * 1.5).iloc[ #indices of troughs in the envelope signal
#             0
#         ][0]
#     )
    
#     onsets = []
#     offsets = []
#     # between two change points 
#     # onset: find the first positive peak that occurs after first change pt
#     # offset: find the last negative peak that occurs before last change pt
#     for i in range(change_points.size - 1):
#         # -- onset calculation
#         p_ix = np.where(pos_peaks > change_points[i])[0] #pos peaks occuring after change point
#         onset_cand = pos_peaks[p_ix[0]]
#         if onset_cand < change_points[i + 1]: # checks if occurs before next change point
#             onset = onset_cand
#         else:
#             onset = np.nan
#         # -- offset calculation
#         n_ix = np.where(neg_peaks < change_points[i + 1])[0] #neg peaks occur after change point
#         offset_cand = neg_peaks[n_ix[-1]]
#         if offset_cand > change_points[i]: # checks if occurs before next change point
#             offset = offset_cand + 3
#         else:
#             offset = np.nan
#         # -- check that onset and offset were calculated
#         test_nan = [onset, offset] #onset and offset lists are created if value returned isnt nan
#         if np.all(~np.isnan(test_nan)):
#             onsets.append(onset)
#             offsets.append(offset)
#     # create a "debug package" that stores any additional information that 
#     # isn't necessarily needed for the function's purpose, but could be 
#     # data or information that could be helpful for diagnostics on the function
#     # that could be useful for modifying parameters
#     debug_pkg = dict()
#     debug_pkg["data"] = dat
#     debug_pkg["envelope"] = env
#     debug_pkg["diff"] = diff.squeeze()
#     debug_pkg["pos_peaks"] = pos_peaks
#     debug_pkg["neg_peaks"] = neg_peaks
#     debug_pkg["change_points"] = change_points 
    
#     return np.array(onsets), np.array(offsets), debug_pkg


##########################################


# plotting ditribution of extensor burst durations, using SL

#calculating burst duration
# #generating onset plots
# names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
#           'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
#           'cat03_013', 'cat03_025', 'cat03_049']


# env_emg_gauss_width_ms = 200  # ms
# gauss_width_ms = 100  # ms
# emg_name = 'emg'
# spk_name = 'spikes'
# smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
# envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
# emg_field = smth_emg_field
# smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"
# pre_idx=400
# post_idx=700

# for name in names:
#     # Load the dataset for the current name
#     nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
#     dataset = NWBDataset(nwb_path)
#     BIN_SIZE = dataset.bin_width

#     # Apply smoothing functions
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

#     # Compute onsets/offsets for the right and left muscles
#     musc_name_right = 'RSL'
#     musc_name_left = 'LSL'

#     r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)
#     l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)


#     # -- refine onsets  -- left 
#     refined_on_l = np.zeros_like(l_ext_on)
#     threshold = 0.2
#     for i, idx_on in enumerate(l_ext_on):    
#         win = l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_on_l[i] = l_ext_on[i] + shift


#     # -- refine onsets  -- right 
#     refined_on_r = np.zeros_like(r_ext_on)
#     threshold = 0.2
#     for i, idx_on in enumerate(r_ext_on):    
#         win = r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_on_r[i] = r_ext_on[i] + shift

#     # -- refine offsets -- left 
#     refined_off_l = np.zeros_like(l_ext_off)
#     threshold = 0.2

#     # onsets left
#     for i, idx_off in enumerate(l_ext_off):    
#         win = l_ext_db_pkg['data'].values[idx_off-pre_idx:idx_off+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) < 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_off_l[i] = l_ext_off[i] + shift


#     #refine offsets -- right 
#     refined_off_r = np.zeros_like(r_ext_off)
#     threshold = 0.2
#     for i, idx_off in enumerate(r_ext_off):    
#         win = r_ext_db_pkg['data'].values[idx_off-pre_idx:idx_off+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) < 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_off_r[i] = r_ext_off[i] + shift


#     for onset, offset in zip(refined_on_r, refined_off_r):
#         burst_durations_r = refined_off_r - refined_on_r
#         burst_durations_l = refined_off_l - refined_on_l


#     fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=200)
#     ax[0].hist(burst_durations_l, bins=20)
#     ax[0].set_title(f"Burst durations for LSL for {name}")

#     ax[1].hist(burst_durations_r, bins=20)
#     ax[1].set_title(f"Burst durations for RSL for {name}")

#     plt.tight_layout()
#     plt.show()

################################################



# # %% -- refine onsets  -- left 

# fig = plt.figure(figsize=(4,6), dpi=150)
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# pre_idx = 500 #changed to 500 for some bc that was the crossing value in the debug code
# post_idx = 700
# refined_on_l = np.zeros_like(l_ext_on)
# threshold = 0.2
# for i, idx_on in enumerate(l_ext_on):    
#     win = l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#     ax.plot(win)
#     cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#     shift = cross_idx - pre_idx
#     refined_on_l[i] = l_ext_on[i] + shift
#     ref_on = refined_on_l[i]
#     ref_win = win = l_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
#     ax2.plot(ref_win)

# axs = [ ax, ax2]

# for ax in axs:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # %% -- refine onsets  -- right 
# fig = plt.figure(figsize=(4,6), dpi=150)

# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# pre_idx = 500
# post_idx = 700
# refined_on_r = np.zeros_like(r_ext_on)
# threshold = 0.2
# for i, idx_on in enumerate(r_ext_on):    
#     win = r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#     ax.plot(win)
#     cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#     shift = cross_idx - pre_idx
#     refined_on_r[i] = r_ext_on[i] + shift
#     ref_on = refined_on_r[i]
#     ref_win = win = r_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
#     ax2.plot(ref_win)

# axs = [ ax, ax2]

# for ax in axs:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# %% -- refine offsets -- left 

# fig = plt.figure(figsize=(4,6), dpi=150)
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# pre_idx = 700
# post_idx = 500
# refined_off_l = np.zeros_like(l_ext_off)
# threshold = 0.2

# for i, idx_off in enumerate(l_ext_off):    
#     win = l_ext_db_pkg['data'].values[idx_off-pre_idx:idx_off+post_idx]
#     ax.plot(win)
#     cross_idx = np.where(np.diff(np.sign(win-threshold)) < 0)[0][0]
#     shift = cross_idx - pre_idx
#     refined_off_l[i] = l_ext_off[i] + shift
#     ref_off = refined_off_l[i]
#     ref_win = win = l_ext_db_pkg['data'].values[ref_off-pre_idx:ref_off+post_idx]
#     ax2.plot(ref_win)

# axs = [ ax, ax2]

# for ax in axs:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # %% -- refine offsets -- right 

# fig = plt.figure(figsize=(4,6), dpi=150)
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# pre_idx = 700
# post_idx = 400
# refined_off_r = np.zeros_like(r_ext_off)
# threshold = 0.2
# for i, idx_off in enumerate(r_ext_off):    
#     win = r_ext_db_pkg['data'].values[idx_off-pre_idx:idx_off+post_idx]
#     ax.plot(win)
#     cross_idx = np.where(np.diff(np.sign(win-threshold)) < 0)[0][0]
#     shift = cross_idx - pre_idx
#     refined_off_r[i] = r_ext_off[i] + shift
#     ref_off = refined_off_r[i]
#     ref_win = win = r_ext_db_pkg['data'].values[ref_off-pre_idx:ref_off+post_idx]
#     ax2.plot(ref_win)

# axs = [ax, ax2]

# for ax in axs:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)



# %%
# generating overview onset/offset plot
# names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
#          'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
#          'cat03_013', 'cat03_025', 'cat03_049'] 

# env_emg_gauss_width_ms = 200  # ms
# gauss_width_ms = 100  # ms
# emg_name = 'emg'
# spk_name = 'spikes'
# smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
# envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
# emg_field = smth_emg_field
# smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"

# for name in names:
#     # Load the dataset for the current name
#     nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
#     dataset = NWBDataset(nwb_path)
#     BIN_SIZE = dataset.bin_width

#     # Apply smoothing functions
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

#     # Compute onsets/offsets for the right muscle
#     musc_name_right = 'RSL'
#     musc_name_left = 'LSL'

#     r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)

#     # Create subplots
#     fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=150)

#     # Plot data on the first subplot
#     amp_diff = r_ext_db_pkg['diff'].values * 100
#     axes[0].plot(r_ext_db_pkg['data'].values, color='k', zorder=3)
#     axes[0].plot(r_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
#     axes[0].plot(r_ext_on, r_ext_db_pkg['data'][r_ext_on], 'o', color='g', markersize=5, zorder=4)
#     axes[0].plot(r_ext_off, r_ext_db_pkg['data'][r_ext_off], 'o', color='r', markersize=5, zorder=4)

#     # Customize subplot appearance
#     axes[0].spines['top'].set_visible(False)
#     axes[0].spines['right'].set_visible(False)

#     axes[0].set_title(f"Raw Signal R side for {name}")

#     l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)


#     amp_diff = l_ext_db_pkg['diff'].values * 100
#     axes[1].plot(l_ext_db_pkg['data'].values, color='k', zorder=3)
#     axes[1].plot(l_ext_db_pkg['envelope'].values, color='g', alpha=0.3)
#     axes[1].plot(l_ext_on, l_ext_db_pkg['data'][l_ext_on], 'o', color='g', markersize=5, zorder=4)
#     axes[1].plot(l_ext_off, l_ext_db_pkg['data'][l_ext_off], 'o', color='r', markersize=5, zorder=4)

#     # Customize subplot appearance
#     axes[1].spines['top'].set_visible(False)
#     axes[1].spines['right'].set_visible(False)

#     axes[1].set_title(f"Raw Signal L side for {name}")

#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()
    
# %%
#generating onset plots
# names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
#          'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
#          'cat03_013', 'cat03_025', 'cat03_049']

# env_emg_gauss_width_ms = 200  # ms
# gauss_width_ms = 100  # ms
# emg_name = 'emg'
# spk_name = 'spikes'
# smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
# envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
# emg_field = smth_emg_field
# smth_spk_name = f"{spk_name}_smooth_{gauss_width_ms}ms"

# for name in names:
#     # Load the dataset for the current name
#     nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
#     dataset = NWBDataset(nwb_path)
#     BIN_SIZE = dataset.bin_width

#     # Apply smoothing functions
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
#     dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=spk_name, name=f"smooth_{gauss_width_ms}ms")

#     # Compute onsets/offsets for the right and left muscles
#     musc_name_right = 'RSL'
#     musc_name_left = 'LSL'

#     r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)
#     l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)

#     # Create subplots
#     fig, axes = plt.subplots(1, 2, figsize=(16, 4), dpi=150)

#     # Refine onsets for the right muscle
#     pre_idx = 500
#     post_idx = 700
#     refined_on_r = np.zeros_like(r_ext_on)
#     if name == 'cat03_043':
#         threshold = 0.3
#     elif name in ['cat03_061', 'cat03_025']:
#         threshold = 0.1
#     else:
#         threshold = 0.2
#     for i, idx_on in enumerate(r_ext_on):    
#         win = r_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_on_r[i] = r_ext_on[i] + shift
#         ref_on = refined_on_r[i]
#         ref_win = r_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
#         axes[0].plot(ref_win, alpha=0.7)  # Plot refined signal

#     axes[0].set_title(f"R Refined Onsets for {name}")
#     axes[0].spines['top'].set_visible(False)
#     axes[0].spines['right'].set_visible(False)

#     # Refine onsets for the left muscle
#     refined_on_l = np.zeros_like(l_ext_on)
#     if name in ['cat03_37', 'cat03_043', 'cat03_047', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_013']:
#         threshold = 0.3
#     else:
#         threshold = 0.2
#     for i, idx_on in enumerate(l_ext_on):    
#         win = l_ext_db_pkg['data'].values[idx_on-pre_idx:idx_on+post_idx]
#         cross_idx = np.where(np.diff(np.sign(win-threshold)) > 0)[0][0]
#         shift = cross_idx - pre_idx
#         refined_on_l[i] = l_ext_on[i] + shift
#         ref_on = refined_on_l[i]
#         ref_win = l_ext_db_pkg['data'].values[ref_on-pre_idx:ref_on+post_idx]
#         axes[1].plot(ref_win, alpha=0.7)  # Plot refined signal

#     axes[1].set_title(f"L Refined Onsets for {name}")
#     axes[1].spines['top'].set_visible(False)
#     axes[1].spines['right'].set_visible(False)

#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()

#refine and visualize muscle activation onsets and offsets for sessions

#################################################################

# # %% --- plot extensor muscle activation over time (SL)
# emg_field = envl_emg_field
# ext_musc_name = 'SL' # BA / MG / SA
# #flx_musc_name = 'TA' # BP / TA / VL
# r_ext_musc_name = f"R{ext_musc_name}"
# l_ext_musc_name = f"L{ext_musc_name}"
# #r_flx_musc_name = f"R{flx_musc_name}"
# #l_flx_musc_name = f"L{flx_musc_name}"
# fig = plt.figure(figsize=(10,4), dpi=300,)
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax.plot(dataset.data[emg_field].index, dataset.data[smth_emg_field][r_ext_musc_name].values)
# ax.plot(dataset.data[emg_field].index, dataset.data[envl_emg_field][r_ext_musc_name].values)
# #ax.plot(dataset.data[emg_field].index, dataset.data[emg_field][r_flx_musc_name].values)

# ax2.plot(dataset.data[emg_field].index, dataset.data[emg_field][l_ext_musc_name].values)
# #ax2.plot(dataset.data[emg_field].index, dataset.data[emg_field][l_flx_musc_name].values)

##############################################

# -- create cache dir if is does not exist
# ds_name = og_ds_name
# if not os.path.isdir(nwb_cache_dir):
#     logger.info(f"Creating {nwb_cache_dir}")
#     os.makedirs(nwb_cache_dir)
# # -- dump dataset object to pickle file
# ds_savepath = os.path.join(nwb_cache_dir, "nwb_" + ds_name + ".pkl")
# # Check if the dataset is valid before attempting to save
# if dataset is None or dataset.data.empty:
#     logger.error(f"Dataset for {ds_name} is None or empty. Skipping save.")
# else:
#     # Construct the save path
#     ds_savepath = os.path.join(nwb_cache_dir, "nwb_" + ds_name + ".pkl")
#     logger.info(f"Saving dataset to {ds_savepath}")

#     try:
#         # Save the dataset to a pickle file
#         with open(ds_savepath, "wb") as rfile:
#             pickle.dump(dataset, rfile, protocol=4)
#             logger.info(f"Dataset {ds_name} saved to pickle.")
#     except Exception as e:
#         logger.error(f"Failed to save dataset {ds_name} to {ds_savepath}: {e}")