# %%  -- package imports

import glob
import os
import _pickle as pickle
import logging
import sys
import numpy as np
from snel_toolkit.datasets.nwb import NWBDataset 
from snel_toolkit.datasets.base import DataWrangler
import pandas as pd
import h5py
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as colormap
from sklearn.decomposition import PCA



# %% -- logger setup
# --- setup logger -- these give more info than print statements
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#### picking side to analyze
SIDE_TO_ANALYZE = "left"

# %% -- iterate through files and load datasets
nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/"

ds_paths = glob.glob(os.path.join(nwb_cache_dir, "*"))


all_ds = []
for ds_path in ds_paths:
    with open(ds_path, 'rb') as file:
        dataset = pickle.load(file)
        all_ds.append(dataset)

emg_gauss_width_ms = 100  # ms
spk_gauss_width_ms = 30 # ms
emg_name = 'emg'
spk_name = 'spike'
smth_spk_name = "spikes_smooth_30ms"  # The prefix for the spike data
num_columns = len(dataset.data['spikes_smooth_30ms'].columns)  # Number of columns in the dataset
spk_names = [(smth_spk_name, i) for i in range(num_columns)]
smth_emg_name = f"emg_smooth_{emg_gauss_width_ms}ms"
spk_names = "spikes"
emg_field = "emg"

smth_spk_name = f"{spk_name}_smooth_{spk_gauss_width_ms}ms"
smth_emg_name = f"{emg_name}_smooth_{emg_gauss_width_ms}ms"

        

# %%
# -- script parameters 
debug = True
save_align_mats = True

# -- analysis params
BIN_SIZE = 10  # ms
GAUSS_WIDTH_MS = 30  # width of gaussian kernel(ms)
CLIP_Q_CUTOFF = 0.99  # emg quantile to set clipping
SCALE_Q_CUTOFF = 0.95  # emg quantile to set unit scaling
XCORR_THRESHOLD = 0.1  # spk/s
ARRAY_SELECT = "ALL"  # which array data to model, if "ALL", use both arrays

NUM_SPK_PCS = 20
NUM_EMG_PCS = 10
L2_SCALE = 1e-2 #1e-0



# %%

# -- paths
align_file_suffix = "_low_reg"
ds_base_name = ds_base_name = sys.argv[1] if len(sys.argv) > 1 else "default_base_name"


# --- create save directories for PCR alignment matrices
base_pcr_save_dir = (
    f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/nwb_lfads/runs/{ds_base_name}/{sys.argv[1]}/"
)
base_align_mat_dir = os.path.join(base_pcr_save_dir, "alignment_matrices")
emg_align_mat_dir = os.path.join(base_align_mat_dir, "emg")
spk_align_mat_dir = os.path.join(base_align_mat_dir, "spikes")


# === end SCRIPT PARAMETERS ==========================

# === begin FUNCTION DEFINITIONS ==========================

# 
def _return_field_names(dataset, field):
    """returns field_names"""
    field_names = dataset.data[field].columns.values.tolist()
    print("Field names: ", field_names)
    return field_names


def aligned_cycle_averaging(dataset_df, dw, field, field_names=None):
    """Perform cycle-aligned averaging."""
    if field_names is None:
        raise ValueError("field_names must be provided.")

    # Ensure field_names match the columns in dw._t_df
    available_columns = dataset_df.columns.tolist()
    field_names = [fname for fname in field_names if fname in available_columns]

    if not field_names:
        raise ValueError(
            f"No matching field names found in dataframe. Available columns: {available_columns}"
        )

    unique_align_times = dataset_df.align_time.unique().size
    num_trials = dataset_df.trial_id.nunique()

    cycle_avg = np.full((unique_align_times, len(field_names)), np.nan)
    cycle_data = np.full(
        (unique_align_times, len(field_names), num_trials),
        np.nan,
    )

    for i, fname in enumerate(field_names):

        # Perform pivot operation
        if fname not in dataset_df.columns:
            raise KeyError(
                f"Column '{fname}' not found in dw._t_df. Available columns: {dw._t_df.columns.tolist()}"
            )

        cycle_aligned_df = dw.pivot_trial_df(dataset_df, values=(fname))
        cycle_avg[:, i] = cycle_aligned_df.mean(axis=1, skipna=True)
        cycle_data[:, i, :] = cycle_aligned_df

    return cycle_avg, cycle_data


def concat_sessions(all_avg, all_means):
    # --- concatenate all sessions together to create "global" space
    global_avg = np.concatenate(all_avg, axis=1)
    global_means = np.concatenate(all_means, axis=1)

    return global_avg, global_means


def fit_global_pcs(global_avg, global_means, num_pcs, fit_ix):

# computing covariance maintenance using pca function
    pca_obj = PCA(n_components=num_pcs)

    # --- mean center, mean centering the data for pcr
    mean_cent_global_avg = global_avg[fit_ix, :] - global_means

    # --- fit pca, eigen decomposition here using pca library
    global_pcs = pca_obj.fit_transform(mean_cent_global_avg)

    return pca_obj, global_pcs


def fit_session_readins(all_avg, all_means, global_pcs, fit_ix, l2_scale=0):
    all_W = []
    all_b = []
    for sess_avg, sess_means in zip(all_avg, all_means):
        lr = Ridge(alpha=l2_scale, fit_intercept=False)
        lr.fit(sess_avg[fit_ix, :] - sess_means, global_pcs)

        # -- weights [ chans x pcs ]
        W = lr.coef_.T

        # -- bias [ pcs ]
        b = sess_means
        # readin layer adds bias after matmul
        b = -1 * np.dot(b, W)
        # print(f"Session Means Shape: {sess_means.shape}")
        # print(f"Weights Shape: {W.shape}")
        # print(f"Bias Shape: {b.shape}")
        # print(f"Bias Values: {b}")
        # print(f"Session {session_id} Spike Means: {sess_means}")
        # print(f"Session {session_id} Spike Weights: {W}")
        all_W.append(W)
        all_b.append(b)

    return all_W, all_b


# === end FUNCTION DEFINITIONS ==========================


# %%

nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/"
ds_paths = glob.glob(os.path.join(nwb_cache_dir, "*"))
logger.info(f"Found {len(ds_paths)} files to process.")

################ step 1: Standardize (zero-mean)
all_spk_chan_means = []
all_spk_cycle_avg = []
all_emg_chan_means = []
all_emg_cycle_avg = []
all_spk_cycle_data = []
all_emg_cycle_data = []

logger.info(f"analyzing {SIDE_TO_ANALYZE}")


for i, dataset in enumerate(all_ds):
    # Filter the data for the current session
    dw = DataWrangler(dataset)
    if SIDE_TO_ANALYZE == "left":
        dataset.trial_info = dataset.l_trial_info
    elif SIDE_TO_ANALYZE == "right":
        dataset.trial_info = dataset.r_trial_info

    
    if 'start_time' not in dataset.trial_info.columns:
        dataset.trial_info['start_time'] = dataset.trial_info['ext_start_time']
    if 'end_time' not in dataset.trial_info.columns:
        dataset.trial_info['end_time'] = dataset.trial_info['ext_stop_time']
    

    dw.make_trial_data(
        name="onset",
        align_field="start_time",
        align_range=(-100, 600),
        allow_overlap=True,
    )
    
    spk_field = "spikes_smooth_30ms"
    emg_field = "emg_smooth_50ms"

    spk_cols = dataset.data[spk_field].columns
    spk_names = [(spk_field, col) for col in spk_cols]

    emg_cols = dataset.data[emg_field].columns
    emg_names = [(emg_field, col) for col in emg_cols]

    if SIDE_TO_ANALYZE == "left":
        emg_names = [name for name in emg_names if name[1].startswith("L")]
    elif SIDE_TO_ANALYZE == "right":
        emg_names = [name for name in emg_names if name[1].startswith("R")]

    # Process spike data
    spk_cycle_avg, spk_cycle_data = aligned_cycle_averaging(
        dw._t_df, dw, spk_field, field_names=spk_names
    )

    spk_chan_means = np.nanmean(spk_cycle_avg, axis=0)[np.newaxis, :]
    all_spk_chan_means.append(spk_chan_means)
    all_spk_cycle_avg.append(spk_cycle_avg)
    all_spk_cycle_data.append(spk_cycle_data)


    # Process EMG data
    emg_cycle_avg, emg_cycle_data = aligned_cycle_averaging(
        dw._t_df, dw, emg_field, field_names=emg_names
    )
    emg_chan_means = np.nanmean(emg_cycle_avg, axis=0)[np.newaxis, :]
    all_emg_chan_means.append(emg_chan_means)
    all_emg_cycle_avg.append(emg_cycle_avg)
    all_emg_cycle_data.append(emg_cycle_data)




#concatenate all session data to create the global data space
global_spk_cycle_avg, global_spk_chan_means = concat_sessions(
    all_spk_cycle_avg, all_spk_chan_means
)
global_emg_cycle_avg, global_emg_chan_means = concat_sessions(
    all_emg_cycle_avg, all_emg_chan_means
)

################### Step 2: covariance matrix
spk_fit_ix = ~np.any(np.isnan(global_spk_cycle_avg), axis=1)
emg_fit_ix = ~np.any(np.isnan(global_emg_cycle_avg), axis=1)

#compute the covariance matrix, compute PCA for spikes
pca_spk, global_spk_pcs = fit_global_pcs(
    global_spk_cycle_avg, global_spk_chan_means, NUM_SPK_PCS, spk_fit_ix
)

emg_fit_ix = ~np.any(np.isnan(global_emg_cycle_avg), axis=1)

#compute PCA for emg
pca_emg, global_emg_pcs = fit_global_pcs(
    global_emg_cycle_avg, global_emg_chan_means, NUM_EMG_PCS, emg_fit_ix
)

all_spk_W, all_spk_b = fit_session_readins(
    all_spk_cycle_avg, all_spk_chan_means, global_spk_pcs, spk_fit_ix, L2_SCALE
)
all_emg_W, all_emg_b = fit_session_readins(
    all_emg_cycle_avg, all_emg_chan_means, global_emg_pcs, emg_fit_ix, L2_SCALE
)
################### step 3: eigen decomposition -- decompose the covariance of the global concatenated dataset


################### step 4: select top-K principal components
# done using NUM_SPK_PCS and NUM_EMG_PCS above

################### Step 5: Project Session Data into Principal Component (PC) Space
session_ids = [os.path.basename(ds_path).split('.')[0] for ds_path in ds_paths]


# results
if save_align_mats:
    # Create directories if they do not exist
    if not os.path.isdir(emg_align_mat_dir):
        logger.info(f"Creating {emg_align_mat_dir}")
        os.makedirs(emg_align_mat_dir)
    if not os.path.isdir(spk_align_mat_dir):
        logger.info(f"Creating {spk_align_mat_dir}")
        os.makedirs(spk_align_mat_dir)

    align_filename = f"pcr_alignment_{SIDE_TO_ANALYZE}{align_file_suffix}.h5"
    lfads_dataset_prefix = "lfads"

    spk_filepath = os.path.join(spk_align_mat_dir, align_filename)
    emg_filepath = os.path.join(emg_align_mat_dir, align_filename)

    with h5py.File(spk_filepath, "w") as hf_spk, h5py.File(emg_filepath, "w") as hf_emg:
        for i, (sess_id, W_spk, b_spk, W_emg, b_emg) in enumerate(zip(session_ids, all_spk_W, all_spk_b, all_emg_W, all_emg_b)):

            ds_name = f"{ds_base_name}_{sess_id}"
            emg_group_name = f"{lfads_dataset_prefix}_{ds_name}_emg_{BIN_SIZE}.h5"
            spk_group_name = f"{lfads_dataset_prefix}_{ds_name}_{ARRAY_SELECT}_spikes_{BIN_SIZE}.h5"

            
            emg_group = hf_emg.create_group(emg_group_name)
            emg_group.create_dataset("matrix", data=W_emg)
            emg_group.create_dataset("bias", data=np.squeeze(b_emg))

            spk_group = hf_spk.create_group(spk_group_name)
            spk_group.create_dataset("matrix", data=W_spk)
            spk_group.create_dataset("bias", data=np.squeeze(b_spk))

        # hf_spk.close()
        # hf_emg.close()


if debug:
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def plot_pcs(ax, pcs, **kwargs):
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], **kwargs)

    cm = colormap.Dark2

    plot_pcs(ax, global_spk_pcs, label="Global", color="k", linewidth=2)
    plot_pcs(ax2, global_emg_pcs, label="Global", color="k", linewidth=2)
    for i, (
        sess_id,
        sess_spk_avg,
        sess_emg_avg,
        sess_spk_data,
        sess_emg_data,
        W_spk,
        b_spk,
        W_emg,
        b_emg,
    ) in enumerate(
        zip(
            session_ids,
            all_spk_cycle_avg,
            all_emg_cycle_avg,
            all_spk_cycle_data,  # <-- Add this
            all_emg_cycle_data,  # <-- Add this
            all_spk_W,
            all_spk_b,
            all_emg_W,
            all_emg_b,
        )
    ):

        session_color = cm(float(i) / len(session_ids))
        
        sess_spk_cycle_pcs = np.matmul(sess_spk_avg[spk_fit_ix, :], W_spk) + b_spk
        plot_pcs(ax, sess_spk_cycle_pcs, label=f"{sess_id} Avg", color=session_color, linewidth=2.5, alpha=0.9)
        
        for trial_idx in range(sess_spk_data.shape[2]):
            trial_activity = sess_spk_data[:, :, trial_idx]
            if not np.all(np.isnan(trial_activity)):
                trial_pcs = np.matmul(trial_activity[spk_fit_ix, :], W_spk) + b_spk
                plot_pcs(ax, trial_pcs, color=session_color, linewidth=0.5, alpha=0.1)
        
        sess_emg_cycle_pcs = np.matmul(sess_emg_avg[emg_fit_ix, :], W_emg) + b_emg
        plot_pcs(ax2, sess_emg_cycle_pcs, label=f"{sess_id} Avg", color=session_color, linewidth=2.5, alpha=0.9)
        
        for trial_idx in range(sess_emg_data.shape[2]):
            trial_activity = sess_emg_data[:, :, trial_idx]
            if not np.all(np.isnan(trial_activity)):
                trial_pcs = np.matmul(trial_activity[emg_fit_ix, :], W_emg) + b_emg
                plot_pcs(ax2, trial_pcs, color=session_color, linewidth=0.5, alpha=0.1)

    #ax.legend()
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"SPIKES ({SIDE_TO_ANALYZE.capitalize()} Side)")
    
    #ax2.legend()
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
    ax2.set_title(f"EMG ({SIDE_TO_ANALYZE.capitalize()} Side)") # Add side to title
    
    plt.suptitle(f"Multi-session PCR Alignment ({SIDE_TO_ANALYZE.capitalize()} Side)") # Add side to suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.figure()
    plt.plot(np.arange(1, len(pca_spk.explained_variance_ratio_) + 1), np.cumsum(pca_spk.explained_variance_ratio_), "-o", color="g", label="spikes")
    plt.plot(np.arange(1, len(pca_emg.explained_variance_ratio_) + 1), np.cumsum(pca_emg.explained_variance_ratio_), "-o", color="b", label="emg")
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Variance Explained by Principal Components")
    #plt.legend()
    plt.grid(True)
    plt.show()


# %%
