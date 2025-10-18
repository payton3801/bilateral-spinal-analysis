import sys
import os
import h5py
import _pickle as pickle
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.base import DataWrangler
from snel_toolkit.interfaces import deEMGInterface, LFADSInterface
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import logging
import yaml
import numpy as np
import pandas as pd

#make sure

# --- setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- handle system inputs
# ds_names = ['cat03_037', 'cat03_039', 'cat03_041', 'cat03_043', 'cat03_045', 'cat03_047', 
#            'cat03_051', 'cat03_053', 'cat03_055', 'cat03_057', 'cat03_059', 'cat03_061',
#            'cat03_013', 'cat03_049']
# for name in ds_names:
#     ds_name = f"cat03_{name}"
ds_name = "cat03"


# === begin SCRIPT PARAMETERS ==========================

lfads_dataset_cfg = [
    {
        "DATASET": {
            "NAME": ds_name,
            "CONDITION_SEP_FIELD": None,  # continuous
            "ALIGN_LIMS": None,
            "ARRAY_SELECT": "R",  # 'R', 'L', 'ALL'
            "BIN_SIZE": 10, # 4,
            "SPK_KEEP_THRESHOLD": None,  # 15,
            "SPK_XCORR_THRESHOLD": 0.1,
            "SPK_MIN_AVG_FR": None,
            "EMG_CLIP_QUANTILE": 0.99,
            "EMG_SCALE_QUANTILE": 0.95,
            "EXCLUDE_TRIALS": [],
            "EXCLUDE_CONDITIONS": [],
            "EXCLUDE_CHANNELS": [],
        }
    },
    {
        "CHOP_PARAMETERS": {
            "TYPE": "emg",
            "DATA_FIELDNAME": "model_emg",
            #"TYPE": "spikes",
            #"DATA_FIELDNAME": "spikes",
            "USE_EXT_INPUT": False,
            "EXT_INPUT_FIELDNAME": "",
            "WINDOW": 1000,  # ms
            "OVERLAP": 200,  # ms
            "MAX_OFFSET": 0,
            "RANDOM_SEED": 0,
            "CHOP_MARGINS": 0,
        }
    },
]


# -- paths
base_name = (
    f"binsize_10ms_pcr_high_reg_{lfads_dataset_cfg[0]['DATASET']['ARRAY_SELECT']}"
)
#ds_base_dir = "/snel/share/share/derived/auyong/NWB/"
lfads_save_dir = '/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/nwb_lfads/runs/datasets'
cache_dir = '/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/'

# === end SCRIPT PARAMETERS ==========================


ds_path = os.path.join(ds_base_dir, ds_name + ".nwb")
# --- load dataset from NWB
logger.info(f"Loading {ds_name} from NWB")
dataset = NWBDataset(ds_path, split_heldout=False)


# --- preprocess spiking data
def generate_spk_keep_chan_mask(dataset, array_select, xcorr_threshold):
    """
    handles selecting spiking channels for modeling based on array selection
    and pairwise corr threshold
    """

    # -- generate mask to select which neurons to include in analysis
    spk_keep_mask = np.array(
        [array_select in val for val in dataset.unit_info.group_name.values]
    )
    if array_select == "ALL":
        spk_keep_mask[:] = True

    if SIDE_TO_ANALYZE == "L":
        dataset.trial_info = dataset.l_trial_info
    elif SIDE_TO_ANALYZE == "R":

        dataset.trial_info = dataset.r_trial_info
    side_neurons = dataset.unit_info.location.apply(lambda x: x.split(' ')[1])
    spk_keep_mask = np.array([SIDE_TO_ANALYZE in side for side in side_neurons]) #make array, spk_keep_mask

    # --- xcorr rejection
    # check that analysis is happening at 1ms
    assert dataset.bin_width == 1
    pair_xcorr, chan_names_to_drop = dataset.get_pair_xcorr(
        "spikes", threshold=xcorr_threshold, zero_chans=True
    )
    # -- update keep mask
    spk_keep_mask[chan_names_to_drop] = False

    return spk_keep_mask


ld_cfg = lfads_dataset_cfg[0]["DATASET"]

# --- drop spk channnels (if necessary)

# -- extract relevant config params
ARRAY_SELECT = ld_cfg["ARRAY_SELECT"]
XCORR_THRESHOLD = ld_cfg["SPK_XCORR_THRESHOLD"]
spk_keep_mask = generate_spk_keep_chan_mask(dataset, ARRAY_SELECT, XCORR_THRESHOLD)

# --- resample dataset (if necessary)

# -- extract relevant config params
BIN_SIZE = ld_cfg["BIN_SIZE"]
if dataset.bin_width != BIN_SIZE:
    logger.info(f"Resampling dataset to bin width (ms): {BIN_SIZE}")
    dataset.resample(BIN_SIZE)


chop_df = dataset.data

# -- drop spk channels ##follow logic from og code
spk_cols = dataset.data[spk_field].columns
spk_cols = spk_cols[spk_mask]
spk_names = [(spk_field, col) for col in spk_cols]
drop_spk_names = spk_names[~spk_keep_mask]

logger.info(f"Keep spike channels: {np.sum(spk_keep_mask)}/{spk_keep_mask.size}")
if type(np.any(drop_spk_names)) == int:
    chop_df.drop(columns=drop_spk_names.tolist(), axis=1, level=1, inplace=True)


# --- preprocess EMG

# -- extract relevant config params
CLIP_Q_CUTOFF = ld_cfg["EMG_CLIP_QUANTILE"]
SCALE_Q_CUTOFF = ld_cfg["EMG_SCALE_QUANTILE"]

clip_emg = chop_df.emg.copy(deep=True)
emg_names = clip_emg.columns.values
# if not list, then expected float
if type(CLIP_Q_CUTOFF) is not list:
    CLIP_Q_CUTOFF = [CLIP_Q_CUTOFF] * emg_names.size

# must pass clip cutoff for each channel if list is passed
assert emg_names.size == len(CLIP_Q_CUTOFF)

for clip_q_cutoff, emg_name in zip(CLIP_Q_CUTOFF, emg_names):
    # -- clipping
    chan_emg = clip_emg[emg_name]
    clip_q = chan_emg.quantile(clip_q_cutoff)
    clip_chan_emg = chan_emg.clip(upper=clip_q)
    # -- scaling
    scale_q = clip_chan_emg.quantile(SCALE_Q_CUTOFF)
    scale_emg = clip_chan_emg / scale_q
    # -- absolute value
    abs_emg = np.abs(scale_emg)
    chop_df[("model_emg", emg_name)] = abs_emg

# --- drop EMG channels (if necessary)
emg_keep_mask = np.ones_like(dataset.data.emg.columns.values, dtype=bool)
drop_emg_names = emg_names[~emg_keep_mask]
logger.info(f"Keep emg channels: {np.sum(emg_keep_mask)}")
if type(np.any(drop_emg_names)) == int:
    chop_df.drop(columns=drop_emg_names.tolist(), axis=1, level=1, inplace=True)
logger.info(f"EMG channel names: {np.array(emg_names)[emg_keep_mask]}")


# --- create save dirs if they do not exist
pkl_dir = os.path.join(lfads_save_dir, "pkls")
if os.path.exists(lfads_save_dir) is not True:
    os.makedirs(lfads_save_dir)
if os.path.exists(pkl_dir) is not True:
    os.makedirs(pkl_dir)

# --- initialize chop interface

chop_cfg = lfads_dataset_cfg[1]["CHOP_PARAMETERS"]

# -- extract relevant config params
WIN_LEN = chop_cfg["WINDOW"]
OLAP_LEN = chop_cfg["OVERLAP"]
MAX_OFF = chop_cfg["MAX_OFFSET"]
CHOP_MARG = chop_cfg["CHOP_MARGINS"]
RAND_SEED = chop_cfg["RANDOM_SEED"]
TYPE = chop_cfg["TYPE"]
NAME = ld_cfg["NAME"]


# setup initial chop fields map (defines which fields will be chopped for lfads)
chop_fields_map = {chop_cfg["DATA_FIELDNAME"]: "data"}

# if we are using external inputs, add this field to chop map
if chop_cfg["USE_EXT_INPUT"]:
    logger.info(
        f"Setting up lfads dataset with external inputs from {chop_cfg['EXT_INPUT_FIELDNAME']}"
    )
    chop_fields_map[chop_cfg["EXT_INPUT_FIELDNAME"]] = "ext_input"

interface = LFADSInterface(
    window=WIN_LEN,
    overlap=OLAP_LEN,
    max_offset=MAX_OFF,
    chop_margins=CHOP_MARG,
    random_seed=RAND_SEED,
    chop_fields_map=chop_fields_map,
)
if TYPE == "emg":
    chan_keep_mask = emg_keep_mask.tolist()
    ds_name = "lfads_" + NAME + "_" + TYPE + "_" + str(BIN_SIZE) + ".h5"
    yaml_name = "cfg_" + NAME + "_" + TYPE + "_" + str(BIN_SIZE) + ".yaml"
    INTERFACE_FILE = os.path.join(
        pkl_dir,
        NAME + "_" + TYPE + "_" + str(BIN_SIZE) + "_interface.pkl",
    )

elif chop_cfg["TYPE"] == "spikes":
    chan_keep_mask = spk_keep_mask.tolist()
    ds_name = (
        "lfads_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + ".h5"
    )
    yaml_name = (
        "cfg_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + ".yaml"
    )
    INTERFACE_FILE = os.path.join(
        pkl_dir,
        NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + "_interface.pkl",
    )

lfads_dataset_cfg[0]["DATASET"]["CHAN_KEEP_MASK"] = chan_keep_mask

# save deemg input and dataset for each session

DATA_FILE = os.path.join(lfads_save_dir, ds_name)
YAML_FILE = os.path.join(lfads_save_dir, yaml_name)

# --- chop and save h5 dataset
interface.chop_and_save(chop_df, DATA_FILE, overwrite=True)

# --- save yaml config file
with open(YAML_FILE, "w") as yamlfile:
    logger.info(f"YAML {YAML_FILE} saved to pickle.")
    data1 = yaml.dump(lfads_dataset_cfg, yamlfile)
    yamlfile.close()

# --- save interface object
with open(INTERFACE_FILE, "wb") as rfile:
    logger.info(f"Interface {INTERFACE_FILE} saved to pickle.")
    pickle.dump(interface, rfile)