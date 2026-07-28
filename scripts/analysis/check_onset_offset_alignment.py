# %%  -- package imports
# using script to test alignment onset and offset and alignment between trials
# # believed that it was too smoothed before and the onset did not align with the real alignments
# # # understand this script lol

import glob
import os
import _pickle as pickle
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from snel_toolkit.datasets.nwb import NWBDataset
import scipy.signal as signal
from scipy.signal import savgol_filter  # Import savgol_filter

# %% -- logger setup
# --- setup logger -- these give more info than print statements
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% -- define paths, locate preproc files

# define path to cache directory
nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/" 

# create wildcard to search that directory (directory + "/*")
ds_paths = glob.glob(os.path.join(nwb_cache_dir, "*"))
print(ds_paths)

# %% -- iterate through files and load datasets

# for loop through each ds path, load in dataset from pickle save to new list called all_ds
all_ds = []
for ds_path in ds_paths:
    with open(ds_path, 'rb') as file:
        logger.info(f"Loading {ds_path}...")
        dataset = pickle.load(file)
        all_ds.append(dataset)
    

def plot_aligned_win(ax, tx, data, pre_idx, post_idx, title=None):
    for t in tx:
        win = data.values[t-pre_idx:t+post_idx]
        ax.plot(win)        
    if title is not None:
        ax.set_title(title)

# %%

env_emg_gauss_width_ms = 100  # ms
gauss_width_ms = 50  # ms
spk_gauss_width_ms = 30 # ms
emg_name = 'emg'
spk_name = 'spikes'

smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
emg_field = smth_emg_field

DS_ID = 0
SIDE = "R" # "L" or "R"

PRE_ALIGN_MS = 0 # ms
POST_ALIGN_MS = 300 # ms


plot_field = "emg_smooth_50ms" # or "spikes_smooth_30ms" or "spikes"
chan_name = "RSL" #"RSL"

# grab the dataset
ds = all_ds[DS_ID]

BIN_SIZE_MS = ds.bin_width

pre_idx = int(PRE_ALIGN_MS/BIN_SIZE_MS)
post_idx = int(POST_ALIGN_MS/BIN_SIZE_MS)


if SIDE == "L":
    tinfo_str = "l_trial_info"
elif SIDE == "R":
    tinfo_str = "r_trial_info"  
else: 
    raise NotImplementedError("Must be R or L.")    

getattr(ds,tinfo_str)
align_times = getattr(ds,tinfo_str).ext_start_time

# get alignment indices
align_idxs = []
for align_time in align_times:
    # append alignment index based on where align time falls in the dataset index
    align_idxs.append(ds.data.index.get_loc(align_time))

#for i, (atime, aidx) in enumerate(zip(align_times, align_idxs)):
    #print(f"{i} -- {atime} --- {aidx}")
for atime in align_times:
    pre_time = atime - pd.to_timedelta(PRE_ALIGN_MS, unit="ms")
    post_time = atime + pd.to_timedelta(POST_ALIGN_MS, unit="ms")
    dat_time = ds.data.loc[pre_time:post_time,:][plot_field][chan_name]
    plt.plot(dat_time.values, color="k", alpha=0.3)
    #dat_idx = ds.data.iloc[aidx-pre_idx:aidx+post_idx,:][plot_field][chan_name]        
    #plt.plot(dat_idx.values)
    
    


# %%
# this plots our current smoothed vs the real data
apt = all_ds[0].l_trial_info.ext_start_time[0]
plt.plot(all_ds[0].data.emg.RSL.values[all_ds[0].data.index.get_loc(apt):all_ds[0].data.index.get_loc(apt)+500])
plt.plot(all_ds[0].data.emg_smooth_100ms.RSL.values[all_ds[0].data.index.get_loc(apt):all_ds[0].data.index.get_loc(apt)+500])




# ------ testing out plotting different thresholds --------
# %%

def compute_on_off_events(ds, musc_name_right , pos_threshold=0.025, neg_threshold=0.03):
    # get data
    dat = ds.data[emg_field][musc_name_right ] #raw signal data
    diff = savgol_filter(dat.values, window_length=5, polyorder=2, deriv=1)  # Adjust window_length and polyorder

    env = ds.data[envl_emg_field][musc_name_right ] #enveloped signal data
    print(f"Data shape: {dat.shape}, Envelope shape: {env.shape}")

    if musc_name_right not in ds.data[emg_field].columns:
        raise KeyError(f"Column '{musc_name_right}' not found in {emg_field}")

    def diff_filter(x): # computes first derivative
        """differentation filter"""
        return signal.savgol_filter(x.to_numpy(), 7, 5, deriv=1, axis=0)

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



ds_names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
           'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
           'cat03_013', 'cat03_025', 'cat03_049']
pre_idx=400
post_idx=700

for name in ds_names:
    nwb_path = f"/snel/share/share/derived/auyong/NWB/{name}.nwb"
    dataset = NWBDataset(nwb_path)
    BIN_SIZE = dataset.bin_width

    # Apply smoothing functions
    dataset.smooth_spk(gauss_width=gauss_width_ms, signal_type=emg_name, name=f"smooth_{gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=env_emg_gauss_width_ms, signal_type=emg_name, name=f"smooth_{env_emg_gauss_width_ms}ms")
    dataset.smooth_spk(gauss_width=spk_gauss_width_ms, signal_type=spk_name, name=f"smooth_{spk_gauss_width_ms}ms")

    # Compute onsets/offsets for the right and left muscles
    musc_name_right = 'RSL'
    musc_name_left = 'LSL'

    r_ext_on, r_ext_off, r_ext_db_pkg = compute_on_off_events(dataset, musc_name_right, pos_threshold=0.001, neg_threshold=0.001)
    l_ext_on, l_ext_off, l_ext_db_pkg = compute_on_off_events(dataset, musc_name_left, pos_threshold=0.001, neg_threshold=0.001)

      # Check if the current dataset is cat03_043
    if name == "cat03_043":
        # Extract the signal for the left muscle (LSL)
        signal = l_ext_db_pkg['data']

        # Loop over all left onset trials for this dataset
        for i, onset_idx in enumerate(l_ext_on):
            compare_thresholds(
                name=name,
                signal=signal,  # Use the extracted signal
                idx=onset_idx,
                thresholds=[0.1, 0.2, 0.3],
                pre_idx=pre_idx,
                post_idx=post_idx,
                title=f"{name} Left Onset Trial {i}"
            )
  
    def refine_tx(tx, data, threshold, pre_idx, post_idx, tx_type="onset"):
        """refine onset/offset calculation"""        
        refined_tx = np.zeros_like(tx)
        data_len = len(data.values)
        for i, idx in enumerate(tx):

            win = data.values[idx-pre_idx:idx+post_idx]
            start_idx = max(0, idx- pre_idx)
            end_idx = (data_len, post_idx + idx)
            if len(win) < pre_idx + post_idx:
                raise ValueError("window size too small for index {idx}. fix pre or post idx")

            if tx_type == "onset":
                cross_pts = np.where(np.diff(np.sign(win-threshold)) > 0)[0]
            elif tx_type == "offset":
                cross_pts = np.where(np.diff(np.sign(win-threshold)) < 0)[0]
            else:
                raise NotImplementedError("tx_type must be onset or offset")
            
            if len(cross_pts) == 0:
                raise ValueError(f"No crossing point found for index {idx} of {name}, check threshold {threshold} and window range {np.min(win)} {np.max(win)}")
            cross_idx = cross_pts[0]
            shift = cross_idx - pre_idx
            refined_tx[i] = tx[i] + shift
        
        return refined_tx
    

# %%
# looking at different threshold values -------------------------------------
def compare_thresholds(name, signal, idx, thresholds = [0.1, 0.2, 0.3], pre_idx=400, post_idx=700, title = ""):
    start = max(0, idx-pre_idx)
    end = min(len(signal), post_idx+idx)
    window = signal[start:end]
    x = np.arange(start, end)

    plt.figure(figsize=(8,4))
    plt.plot(x, window, label="signal")

    for threshold in thresholds:
        crossing = np.where(window > threshold)[0]
        if len(crossing) > 0:
            cross_idx = start + crossing[0]
            plt.axhline(threshold, linestyle="--", label=f"threshold={threshold}", alpha=0.7)
            plt.axvline(cross_idx, linestyle=":", alpha=0.7)

    plt.legend()
    plt.show()
    
# %%




# loop over all left onset trials for this dataset
for i, onset_idx in enumerate(l_ext_on):
    compare_thresholds(
        name = "cat03_043",
        signal=l_ext_db_pkg['data'], 
        idx=onset_idx,               
        thresholds=[0.1, 0.2, 0.3],
        pre_idx=400,
        post_idx=700,
        title=f"{name} Left Onset Trial {i}"
    )


# %%
