# %%  -- package imports
import glob
import os
import _pickle as pickle
import logging
import sys
import numpy as np

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
        dataset = pickle.load(file)
        all_ds.append(dataset)


# %%
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

model_emg_field = "model_emg"

# -- paths
align_file_suffix = "_low_reg"
base_name = f"binsize_10ms_pcr_high_reg_{ARRAY_SELECT}"

# --- create save directories for PCR alignment matrices
base_pcr_save_dir = (
    f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/nwb_lfads/runs/{base_name}/{sys.argv[1]}/"
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
    return field_names


def aligned_cycle_averaging(dataset, dw, field, field_names=None):
    """perform cycle aligned averaging"""
    if field_names is None:
        field_names = _return_field_names(ds, field)
    cycle_avg = np.full((dw._t_df.align_time.unique().size, len(field_names)), np.nan)
    cycle_data = np.full(
        (
            dw._t_df.align_time.unique().size,
            len(field_names),
            dw._t_df.trial_id.nunique(),
        ),
        np.nan,
    )

    for i, fname in enumerate(field_names):
        cycle_aligned_df = dw.pivot_trial_df(dw._t_df, values=(field, fname))
        cycle_avg[:, i] = cycle_aligned_df.mean(axis=1, skipna=True)
        cycle_data[:, i, :] = cycle_aligned_df
    return cycle_avg, cycle_data


def concat_sessions(all_avg, all_means):
    # --- concatenate all sessions together to create "global" space
    global_avg = np.concatenate(all_avg, axis=1)
    global_means = np.concatenate(all_means, axis=1)

    return global_avg, global_means


def fit_global_pcs(global_avg, global_means, num_pcs, fit_ix):
    from sklearn.decomposition import PCA

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
        all_W.append(W)
        all_b.append(b)

    return all_W, all_b


# === end FUNCTION DEFINITIONS ==========================

# %%

spk_gauss_width_ms = 30 # ms
emg_gauss_width_ms = 30 # ms
emg_name = 'emg'
spk_name = 'spikes'

smth_spk_name = f"{spk_name}_smooth_{spk_gauss_width_ms}ms"
smth_emg_name = f"{emg_name}_smooth_{emg_gauss_width_ms}ms"

# %%
env_emg_gauss_width_ms = 100  # ms
gauss_width_ms = 50  # ms
spk_gauss_width_ms = 30 # ms
emg_name = 'emg'
spk_name = 'spikes'


################ step 1: Standardize (zero-mean)
spk_cycle_avg, _ = aligned_cycle_averaging(
    dataset, dw, spk_field, field_names=spk_names
)
spk_chan_means = np.nanmean(spk_cycle_avg, axis=0)[np.newaxis, :]
all_spk_chan_means.append(spk_chan_means)
all_spk_cycle_avg.append(spk_cycle_avg)

emg_cycle_avg, _ = aligned_cycle_averaging(
    dataset, dw, emg_field, field_names=emg_names
)
emg_chan_means = np.nanmean(emg_cycle_avg, axis=0)[np.newaxis, :]
all_emg_chan_means.append(emg_chan_means)
all_emg_cycle_avg.append(emg_cycle_avg)

#concatenate all session data to create the global data space
global_spk_cycle_avg, global_spk_chan_means = concat_sessions(
    all_spk_cycle_avg, all_spk_chan_means
)
global_emg_cycle_avg, global_emg_chan_means = concat_sessions(
    all_emg_cycle_avg, all_emg_chan_means
)

################### Step 2: covariance matrix
spk_fit_ix = ~np.any(np.isnan(global_spk_cycle_avg), axis=1)

#compute the covariance matrix, compute PCA for spikes
pca_spk, global_spk_pcs = fit_global_pcs(
    global_spk_cycle_avg, global_spk_chan_means, NUM_SPK_PCS, spk_fit_ix
)

emg_fit_ix = ~np.any(np.isnan(global_emg_cycle_avg), axis=1)

#compute PCA for emg
pca_emg, global_emg_pcs = fit_global_pcs(
    global_emg_cycle_avg, global_emg_chan_means, NUM_EMG_PCS, spk_fit_ix
)
################### step 3: eigen decomposition -- decompose the covariance of the global concatenated dataset


################### step 4: select top-K principal components
# done using NUM_SPK_PCS and NUM_EMG_PCS above

################### Step 5: Project Session Data into Principal Component (PC) Space

# results
if save_align_mats:
    # Create directories if they do not exist
    if not os.path.isdir(emg_align_mat_dir):
        logger.info(f"Creating {emg_align_mat_dir}")
        os.makedirs(emg_align_mat_dir)
    if not os.path.isdir(spk_align_mat_dir):
        logger.info(f"Creating {spk_align_mat_dir}")
        os.makedirs(spk_align_mat_dir)

    align_filename = f"pcr_alignment{align_file_suffix}.h5"
    hf_spk = h5py.File(os.path.join(spk_align_mat_dir, align_filename), "w")
    hf_emg = h5py.File(os.path.join(emg_align_mat_dir, align_filename), "w")

    for i, (sess_id, W_spk, b_spk, W_emg, b_emg) in enumerate(
        zip(
            session_ids,
            all_spk_W,
            all_spk_b,
            all_emg_W,
            all_emg_b,
        )
    ):
        emg_group_name = (
            "_".join((lfads_dataset_prefix, ds_name, "emg", str(BIN_SIZE))) + ".h5"
        )
        spk_group_name = (
            "_".join(
                (lfads_dataset_prefix, ds_name, ARRAY_SELECT, "spikes", str(BIN_SIZE))
            )
            + ".h5"
        )
        emg_group = hf_emg.create_group(emg_group_name)
        emg_group.create_dataset("matrix", data=W_emg)
        emg_group.create_dataset("bias", data=np.squeeze(b_emg))

        spk_group = hf_spk.create_group(spk_group_name)
        spk_group.create_dataset("matrix", data=W_spk)
        spk_group.create_dataset("bias", data=np.squeeze(b_spk))

    hf_spk.close()
    hf_emg.close()

# %%
