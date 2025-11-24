#%%
# import rds
import sys
import pdb

# from rds.datasets.xds import XDSDataset, get_rds_dataset
# from rds.interfaces import deEMGInterface
# from rds.decoding import NeuralDecoder, prepare_decoding_data
# from rds.structures import DataWrangler
from scipy import signal
from os import path
import os
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import yaml
from lfads_tf2.utils import load_posterior_averages
from snel_toolkit.datasets.nwb import NWBDataset

#from nlb_tools.nwb_interface import NWBDataset

# -- setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

## -- paths to necessary pbt directories to load output
#
# base_name = "binsize_10ms_all_sess"
# base_name = "binsize_10ms_post_emg_name_fix"
# base_name = "binsize_4ms_ALL"
#base_name = "binsize_4ms_pcr_high_reg_ALL"
BASE_NAME = "binsize_10ms_pcr_high_reg_R"
use_cached = True

# nwb_cache_dir = f"/snel/share/share/tmp/scratch/lwimala/auyong_nwb/{base_name}/"
ds_base_dir = "/snel/share/share/tmp/scratch/pbechef"

ds_base_name = "cat03"
# session_ids = sys.argv[2]
# session_ids = session_ids.split(",")
# session_ids[0] = session_ids[0].replace("[", "")
# session_ids[-1] = session_ids[-1].replace("]", "")

session_ids = [
    "013_R", "037_R", "039_R", "041_R", "043_R", "045_R",
    "047_R", "049_R", "051_R", "053_R", "055_R", "057_R",
    "059_R", "061_R"
]

# ms_prefix = "low_jit_"
# ms_prefix = "high_reg_"
#ms_prefix = "low_con_"
#ms_prefix = "low_reg_"
ms_prefix = ""
run_type = "emg"
#run_type = "spikes"
RUN_TYPE = "spikes"
RUN_DIR = "run_001/"


RUN_HOME = f"/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/nwb_lfads/runs/{BASE_NAME}"
# lfads_dataset_dir = f"/snel/share/share/derived/auyong/nwb_lfads/runs/{base_name}/{ds_base_name}/datasets/"
PBT_HOME = path.join(RUN_HOME, f"pbt_runs/{RUN_TYPE}/{RUN_DIR}/")
lfads_dataset_dir = path.join(RUN_HOME, f"pbt_runs/{RUN_TYPE}/lfads_input")
model_dir = path.join(PBT_HOME, "pbt_run/best_model")
# base_name = "binsize_4ms_ALL"
#nwb_cache_dir = f"/snel/share/share/tmp/scratch/pbechef/"
nwb_cache_dir = '/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/preprocessed/'
## -- load sampling output
ps_filename = "posterior_samples.h5"
sampling_output = load_posterior_averages(
    model_dir, merge_tv=True, ps_filename=ps_filename
)
## -- merge output for each session
for session_id in session_ids:
    # get datset name
    ds_name = f"{ds_base_name}_{session_id}"
    if use_cached:
        ds_path = path.join(nwb_cache_dir, "nwb_" + ds_name + ".pkl")
        logger.info(f"Loading {ds_name} from pickled NWB")
        with open(ds_path, "rb") as rfile:
            dataset = pickle.load(rfile)
            logger.info("Dataset loaded from pickle.")

    else:
        ds_path = path.join(ds_base_dir, ds_name + ".nwb")
        logger.info("Loading from NWB")
        dataset = NWBDataset(ds_path)
        # if needed, resample
        dataset.resample(BIN_SIZE)
    print(dataset.data.spikes.shape)
    yaml_name = f"cfg_{ds_name}*.yaml"
    # interface_name = f"pkls/{ds_name}*interface.pkl"
    interface_name = f"interface*{ds_name}_spikes_10.pkl"
    h5_name = f"*{ds_name}*.h5"
    cfg_yaml_filepath = glob.glob(os.path.join(lfads_dataset_dir, yaml_name))[0]
    interface_filepath = glob.glob(os.path.join(lfads_dataset_dir, interface_name))[0]
    lfads_ds_filepath = glob.glob(os.path.join(lfads_dataset_dir, h5_name))[0]
    lfads_ds_filename = lfads_ds_filepath.split("/")[-1]  # get name of h5
    lfads_dataset_name = lfads_ds_filename.replace("lfads_", "")  # remove prefix

    ## -- yaml config loading
    with open(cfg_yaml_filepath, "r") as yamlfile:
        logger.info("Loading YAML config node")
        cfg_node = yaml.load(yamlfile, Loader=yaml.FullLoader)
        ## -- unpack needed parameters from config
    BIN_SIZE = cfg_node[0]["DATASET"]["BIN_SIZE"]
    DATASET_NAME = cfg_node[0]["DATASET"]["NAME"]
    ARRAY_SELECT = cfg_node[0]["DATASET"]["ARRAY_SELECT"]
    TYPE = cfg_node[1]["CHOP_PARAMETERS"]["TYPE"]

    DATA_FIELDNAME = cfg_node[1]["CHOP_PARAMETERS"]["DATA_FIELDNAME"]
    print(DATA_FIELDNAME)
    with open(interface_filepath, "rb") as rfile:
        interface = pickle.load(rfile)
        logger.info("LFADS Interface loaded from pickle.")

    base_spikes_load_names = ["lfads_rates", "lfads_factors", "lfads_gen_inputs"]
    base_emg_load_names = [
        "deEMG_mean",
        "deEMG_var",
        "deEMG_factors",
        "deEMG_gen_inputs",
        ]

    emg_load_names = []
    spikes_load_names = []
    for s_load_name in base_spikes_load_names:
        spikes_load_names.append(ms_prefix + s_load_name)
    for e_load_name in base_emg_load_names:
        emg_load_names.append(ms_prefix + e_load_name)
    # build data_dict
    data_dict = {}
    data_dict["factors"] = sampling_output[lfads_dataset_name].factors
    data_dict["gen_inputs"] = sampling_output[lfads_dataset_name].gen_inputs

        # update merge fields map and add output params to data dict
        # need to handle spikes/emg separately since output params have different shapes
    if TYPE == "spikes":
        interface.merge_fields_map = {
            "output_params": spikes_load_names[0],
            "factors": spikes_load_names[1],
            "gen_inputs": spikes_load_names[2],
            }
        data_dict["output_params"] = np.squeeze(
            sampling_output[lfads_dataset_name].output_params
            )
        load_names = spikes_load_names
        match_col_names = [True, False, False]
    elif TYPE == "emg":
        interface.merge_fields_map = {
            "output_par_1": emg_load_names[0],
            "output_par_2": emg_load_names[1],
            "factors": emg_load_names[2],
            "gen_inputs": emg_load_names[3],
            }
        data_dict["output_par_1"] = np.squeeze(
            sampling_output[lfads_dataset_name].output_params[:, :, :, 0]
            )
        data_dict["output_par_2"] = np.squeeze(
            sampling_output[lfads_dataset_name].output_params[:, :, :, 1]
            )
        load_names = emg_load_names
        match_col_names = [True, True, False, False]

        # merge lfads output to continuous-like form
    cts_df = interface.merge(data_dict, smooth_pwr=1)
    chan_keep_mask = cfg_node[0]["DATASET"]["CHAN_KEEP_MASK"]
    input_field = DATA_FIELDNAME
    MO_FIELDS = load_names

    MATCH_FIELDNAMES = match_col_names
    for MO_FIELD, MATCH_FIELDNAME in zip(MO_FIELDS, MATCH_FIELDNAMES):
        if MATCH_FIELDNAME:
            chan_names = dataset.data[input_field].columns.values
            #import pdb; pdb.set_trace()

            chan_names = chan_names[chan_keep_mask]
        else:
            chan_names = None
            # drop columns if they are already in the dataset before adding
        if MO_FIELD in dataset.data.columns.levels[0]:
            logger.warning(f"Overwriting {MO_FIELD} in dataframe")
            dataset.data = dataset.data.drop(
                [MO_FIELD], level=0, axis=1
                )  # errors='ignore'
        else:
            logger.info(f"Adding {MO_FIELD} in dataframe")
            dataset.add_continuous_data(
                cts_df[MO_FIELD].values, MO_FIELD, chan_names=chan_names
            )

        # drop duplicated columns
    dataset.data = dataset.data.loc[:, ~dataset.data.columns.duplicated()]
    ds_savepath = path.join(nwb_cache_dir, "nlb_" + DATASET_NAME + ".pkl")
    with open(ds_savepath, "wb") as rfile:
        logger.info(f"Dataset {DATASET_NAME} saved to pickle.")
        pickle.dump(dataset, rfile, protocol=4)

# import pdb

# pdb.set_trace()


# %%
