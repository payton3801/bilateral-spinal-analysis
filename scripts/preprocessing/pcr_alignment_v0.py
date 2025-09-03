# %%  -- package imports
import glob
import os
import _pickle as pickle
import logging
import sys

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
# -- script control options from https://github.com/snel-repo/emg-dynamics/blob/06365573e192207ee8dd31a10cbfce6abf594e5b/emg_paper/nwb_conversion/auyong_pcr_alignment_v4.py#L61-L149
debug = True
save_align_mats = False

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
    f"/snel/share/share/derived/auyong/nwb_lfads/runs/{base_name}/{sys.argv[1]}/"
)
base_align_mat_dir = os.path.join(base_pcr_save_dir, "alignment_matrices")
emg_align_mat_dir = os.path.join(base_align_mat_dir, "emg")
spk_align_mat_dir = os.path.join(base_align_mat_dir, "spikes")


# === end SCRIPT PARAMETERS ==========================

# === begin FUNCTION DEFINITIONS ==========================
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

    pca_obj = PCA(n_components=num_pcs)

    # --- mean center
    mean_cent_global_avg = global_avg[fit_ix, :] - global_means

    # --- fit pca
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
