# %%
#see proprocss step info

from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.base import DataWrangler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import signal
from snel_toolkit.datasets.base import DataWrangler

import glob
import os
import _pickle as pickle
import logging
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors



# %%
# --- setup logger -- these give more info than print statements
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%

base_path = "/snel/share/share/derived/auyong/NWB/" 
nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/" 


ds_names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
           'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
           'cat03_013', 'cat03_049']  #cat03_025 being a pain

#saving this preprocessed data
if not os.path.exists(nwb_cache_dir):
    os.makedirs(nwb_cache_dir)

env_emg_gauss_width_ms = 100  # ms
gauss_width_ms = 50  # ms
spk_gauss_width_ms = 30 # ms
emg_name = 'emg'
spk_name = 'spikes'

smth_emg_field = f"{emg_name}_smooth_{gauss_width_ms}ms"
envl_emg_field = f"{emg_name}_smooth_{env_emg_gauss_width_ms}ms"
emg_field = smth_emg_field
#smth_spk_name = f"{spk_name}_smooth_{spk_gauss_width_ms}ms"

# %%
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
## need to figure out why alignment is not working, could be refinement values or threshold 
# breaks on second iteration
# we decreased smoothing

pre_idx=800
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
                raise ValueError(f"No crossing point found for index {idx} of {name}, check threshold {threshold} and window min {np.min(win)}")
            cross_idx = cross_pts[0]
            shift = cross_idx - pre_idx
            refined_tx[i] = tx[i] + shift
        
        return refined_tx
    
    def plot_aligned_win(ax, tx, data, pre_idx, post_idx, title=None):
        cmap = cm.get_cmap('rainbow', len(tx))  
        
        for i, t in enumerate(tx):
            win = data.values[t-pre_idx:t+post_idx]
            color = cmap(i)   # pick distinct color
            ax.plot(win, color=color, label=f"Index: {t}")      
        if title is not None:
            ax.set_title(title)
    # Create subplots
        ax.legend(loc = "upper right", fontsize = "small")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

   ## -- refine onsets -- left 
    if name in ['cat03_039', 'cat03_043', "cat03_013"]:
        threshold = 0.4
    elif name == 'cat03_025':
        threshold = 1.1

    else:
        threshold = 0.2  
  
    refined_on_l = refine_tx(l_ext_on, l_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="onset")

    # -- refine onsets -- right
    if name == 'cat03_037':
        threshold = 0.5
    elif name in ['cat03_043', 'cat03_025']:
        threshold = 0.15
    elif name == "cat03_013":
        threshold = 0.4
    
    else:
        threshold = 0.3
    refined_on_r = refine_tx(r_ext_on, r_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="onset")

    # -- refine offsets -- left    
    if name in ['cat03_039', 'cat03_043', 'cat03_025']:
        threshold = 0.3
    elif name == "cat03_013":
        threshold = 0.4
    else:
        threshold = 0.2
    #break
    refined_off_l = refine_tx(l_ext_off, l_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="offset")
    # -- refine offsets -- right 
    refined_off_r = refine_tx(r_ext_off, r_ext_db_pkg['data'], threshold, pre_idx, post_idx, tx_type="offset")

  # indices removal
    indices_to_remove = {
        'cat03_037': {
        'left':[29627],
        'right':[21834, 16832, 30223, 24182], 
        },
        'cat03_039': {
        'left':[7516],
        'right':[19415], 
        },
        'cat03_041': {
        'left':[10691, 24696],
        },
        'cat03_045': {
        'left':[34052, 26286, 35428],
        'right':[34771, 29191], 
        },
        'cat03_047': {
        'right':[16110, 24067], 
        },
        'cat03_051': {
        'right':[21493, 18736], 
        },
        'cat03_053': {
        'left':[27205, 22684, 17929, 24548],
        'right':[24299], 
        },
        'cat03_055': {
        'left':[11447],
        'right':[26498], 
        },
        'cat03_057': {
        'left':[17199],
        'right':[ 30934], 
        },
        'cat03_059': {
        'left':[11295, 14552],
        'right':[16230, 16172], 
        },
        'cat03_061': {
        'left':[20895, 15288],
        'right':[15375, 27091], 
        },
        'cat03_013': {
        'left':[11637],
        },
        'cat03_049': {
        'left':[25060],
        'right':[18550, 18230, 24546, 18484, 24386], 
        },
    }

        
    if name in indices_to_remove:
        remove_indices_left = indices_to_remove[name].get('left', [])
        remove_indices_right = indices_to_remove[name].get('right', [])
        remove_indices = indices_to_remove[name]

        mask_on_l = ~np.isin(refined_on_l, remove_indices_left)
        mask_off_l = ~np.isin(refined_off_l, remove_indices_left)
        
        # Apply the masks to remove the indices for left side
        refined_on_l = refined_on_l[mask_on_l]
        refined_off_l = refined_off_l[mask_off_l]
        
        mask_on_r = ~np.isin(refined_on_r, remove_indices_right)
        mask_off_r = ~np.isin(refined_off_r, remove_indices_right)
        
        refined_on_r = refined_on_r[mask_on_r]
        refined_off_r = refined_off_r[mask_off_r]


    plot_aligned_win(axes[0,0], refined_on_l, l_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} L Onset")
    plot_aligned_win(axes[0,1], refined_on_r, r_ext_db_pkg['data'], pre_idx, post_idx, title=f"{name} R Onset")
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
        logger.info(f"Saving {pkl_save_path} to pickle.")
        pickle.dump(dataset, f)


# %%
## plotting flexor and extensor muscle plots
for name in ds_names:
    
    save_filename = f"{name}_preproc.pkl"
    pkl_save_path = os.path.join(nwb_cache_dir, save_filename)

    with open(pkl_save_path, 'rb') as f:
        dataset = pickle.load(f)

    ext = dataset.data[emg_field]["RBA"].values
    flex = dataset.data[emg_field]["LSL"].values

    refined_on_r = [dataset.data.index.get_loc(ts) for ts in dataset.r_trial_info['ext_start_time'].values]
    refined_off_r = [dataset.data.index.get_loc(ts) for ts in dataset.r_trial_info['ext_stop_time'].values]

    flex_shift = ext + 2
    x = np.arange(len(ext))


    plt.figure(figsize=(12,5))
    plt.plot(x, ext, color="orange", label = "extensor")
    plt.plot(x, flex + flex_shift, color = "blue", label = "flexor")

    on_x = [ix for ix in refined_on_r if 0 <= ix < len(x)]
    off_x = [ix for ix in refined_off_r if 0 <= ix < len(x)]

    plt.vlines(on_x, ymin = np.nanmin(ext), ymax= np.nanmax(flex + flex_shift), color = "g", linestyle = "--", alpha = 0.6, label = "Onset")
    plt.vlines(off_x, ymin = np.nanmin(ext), ymax = np.nanmax(flex + flex_shift), color = "b", linestyle = "--", alpha = 0.6, label = "Offset")

    plt.title(f"{name} Muscle activity w/ onset and offset")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
