# %%
import sys
import os
import glob
import pandas as pd
import numpy as np
import pickle
import logging
import yaml
from os import path
from lfads_tf2.utils import load_posterior_averages
from snel_toolkit.datasets.nwb import NWBDataset

# -- Setup Logging --
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

ds_base_dir = "/snel/share/share/tmp/scratch/bilateral_cat"
ds_base_name = "cat03"
nwb_cache_dir = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache'
step_info_dir = path.join(nwb_cache_dir, "step_info") 
output_dir = path.join(nwb_cache_dir, "merged_datasets") 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

session_ids = [
    "013", "037", "039", "041", "043", "045",
    "047", "049", "051", "053", "055", "057",
    "059", "061"
]

RUN_CONFIGS = [
    {"name": "binsize_10ms_pcr_L",   "type": "spikes", "suffix": "_L"},
    {"name": "binsize_10ms_pcr_R",   "type": "spikes", "suffix": "_R"},
    {"name": "binsize_10ms_pcr_ALL", "type": "emg",    "suffix": "_EMG"},
]

RUN_DIR_NAME = "run_001/"

loaded_models = {}

for config in RUN_CONFIGS:
    base_name = config["name"]
    logger.info(f"Loading model output for: {base_name}")
    
    # Path logic
    run_home = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{base_name}"
    pbt_home = path.join(run_home, f"pbt_runs/{config['type']}/{RUN_DIR_NAME}/")
    model_dir = path.join(pbt_home, "pbt_run/best_model")
    

    sampling_output = load_posterior_averages(model_dir, merge_tv=True, ps_filename="posterior_samples.h5")
    loaded_models[base_name] = {
        "data": sampling_output,
        "lfads_input_dir": path.join(run_home, f"pbt_runs/{config['type']}/lfads_input")
    }


def apply_step_info(dataset, ds_name):
    """Loads l_trial_info/r_trial_info and attaches it to the dataset object."""
    info_path = path.join(step_info_dir, f"{ds_name}_step_info.pkl")

    with open(info_path, "rb") as f:
        step_data = pickle.load(f)

    if hasattr(step_data, 'l_trial_info'): dataset.l_trial_info = step_data.l_trial_info
    if hasattr(step_data, 'r_trial_info'): dataset.r_trial_info = step_data.r_trial_info
    if hasattr(step_data, 'trial_info'): dataset.trial_info = step_data.trial_info
    

def safe_add_data(dataset, data_block, save_name, chan_names=None):
    """adds continuous data to the dataset"""
    if isinstance(dataset.data.columns, pd.MultiIndex):
        if save_name in dataset.data.columns.get_level_values(0):
            dataset.data = dataset.data.drop(save_name, axis=1, level=0)
    elif save_name in dataset.data.columns:
        dataset.data = dataset.data.drop(save_name, axis=1)
    
    # if channel names not woring
    if chan_names is None:
        chan_names = list(map(str, range(data_block.shape[1])))
    
    dataset.add_continuous_data(
        data_block, 
        save_name, 
        chan_names=chan_names 
    )

for session_id in session_ids:
    ds_name = f"{ds_base_name}_{session_id}"
    logger.info(f"--- Processing Session: {session_id} ---")
    
    ds_path = path.join(nwb_cache_dir, "post_pcr", ds_name + "_post_pcr.pkl")
    if path.exists(ds_path):
        with open(ds_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        raw_path = path.join(ds_base_dir, ds_name + ".nwb")
        dataset = NWBDataset(raw_path)

    apply_step_info(dataset, ds_name)

    for config in RUN_CONFIGS:
        run_name = config["name"]
        suffix = config["suffix"]
        is_emg = (config["type"] == "emg")
        
        if run_name not in loaded_models:
            continue
            
        model_pack = loaded_models[run_name]
        sampling_output = model_pack["data"]
        lfads_input_dir = model_pack["lfads_input_dir"]
        
        h5_pattern = f"*{ds_name}*.h5"

        lfads_path = glob.glob(os.path.join(lfads_input_dir, h5_pattern))[0]
        lfads_key = lfads_path.split("/")[-1].replace("lfads_", "")


        interface_pattern = f"{ds_name}_*_emg_10_interface.pkl" if is_emg else f"{ds_name}_*_spikes_10_interface.pkl"
        try:
            int_path = glob.glob(os.path.join(lfads_input_dir, interface_pattern))[0]
            with open(int_path, "rb") as f: interface = pickle.load(f)
        except IndexError:
            logger.warning(f"   Skipping {run_name}: Interface not found.")
            continue

        data_dict = {
            "factors": sampling_output[lfads_key].factors,
            "gen_inputs": sampling_output[lfads_key].gen_inputs
        }

        if is_emg:
            names = ["deEMG_mean", "deEMG_var", "deEMG_factors", "deEMG_gen_inputs"]
            interface.merge_fields_map = {"output_par_1": names[0], "output_par_2": names[1], "factors": names[2], "gen_inputs": names[3]}
            data_dict["output_par_1"] = np.squeeze(sampling_output[lfads_key].output_params[:,:,:,0])
            data_dict["output_par_2"] = np.squeeze(sampling_output[lfads_key].output_params[:,:,:,1])
        else:
            names = ["lfads_rates", "lfads_factors", "lfads_gen_inputs"]
            interface.merge_fields_map = {"output_params": names[0], "factors": names[1], "gen_inputs": names[2]}
            data_dict["output_params"] = np.squeeze(sampling_output[lfads_key].output_params)

        cts_df = interface.merge(data_dict, smooth_pwr=1)

        if isinstance(cts_df.columns, pd.MultiIndex):
            unique_signals = cts_df.columns.get_level_values(0).unique()
            for signal_name in unique_signals:
                data_block = cts_df[signal_name].values
                if data_block.ndim == 1: data_block = data_block.reshape(-1, 1)
                
                safe_add_data(dataset, data_block, f"{signal_name}{suffix}")
        else:
            for col in cts_df.columns:
                data_block = cts_df[col].values
                if data_block.ndim == 1: data_block = data_block.reshape(-1, 1)
                safe_add_data(dataset, data_block, f"{col}{suffix}")
        
    final_save_path = path.join(output_dir, f"merged_{ds_name}.pkl")
    
    with open(final_save_path, "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    

# %%
#putting all sessions in ds_all

input_dir = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache/merged_datasets'

session_ids = [
    "013", "037", "039", "041", "043", "045",
    "047", "049", "051", "053", "055", "057",
    "059", "061"
]


all_ds = {}

print("Loading sessions into memory...")

for sid in session_ids:

    filename = f"merged_cat03_{sid}.pkl"
    file_path = os.path.join(input_dir, filename)
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            all_ds[sid] = pickle.load(f)
            print(f"Loaded Session {sid}")
    else:
        print(f"File not found for {sid}")

print(f"Loaded {len(all_ds)} datasets into 'all_ds'.")

# %%
# look at individual sessions

target_sess = "061"

ds = all_ds[target_sess]

if hasattr(ds, 'l_trial_info'):
    display(ds.l_trial_info.head())

print("\nFirst 5 columns in this dataset:")
print(ds.data.columns[:5].tolist())


# %%
