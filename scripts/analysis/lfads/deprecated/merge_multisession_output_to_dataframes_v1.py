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

# -- setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
run_list = [
    "binsize_10ms_pcr_L", 
    "binsize_10ms_pcr_R",
    "binsize_10ms_pcr_ALL",   
]

ds_base_dir = "/snel/share/share/tmp/scratch/bilateral_cat"
ds_base_name = "cat03"
nwb_cache_dir = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_cache'

session_ids = [
    "013", "037", "039", "041", "043", "045",
    "047", "049", "051", "053", "055", "057",
    "059", "061"
]

ms_prefix = ""
use_cached = True
RUN_DIR = "run_001/"

# ---------------------------------------------------------------------------
# PROCESSING LOOP
# ---------------------------------------------------------------------------
for BASE_NAME in run_list:
    logger.info(f"==========================================")
    logger.info(f"PROCESSING RUN: {BASE_NAME}")
    logger.info(f"==========================================")

    # 1. Detect Run Type (Spikes vs EMG)
    # Logic: If it contains 'pcr_ALL', it is your EMG run
    if "pcr_all" in BASE_NAME.lower():
        IS_EMG = True
        current_run_type_folder = "emg"  
        name_mod_suffix = "_emg"
        logger.info(f"-> EMG Run Detected (Folder: {current_run_type_folder})")
    elif "emg" in BASE_NAME.lower():
        IS_EMG = True
        current_run_type_folder = "emg"  
        name_mod_suffix = "_emg"
        logger.info(f"-> EMG Run Detected (Folder: {current_run_type_folder})")
    else:
        IS_EMG = False
        current_run_type_folder = "spikes"
        name_mod_suffix = ""
        logger.info(f"-> Spikes Run Detected (Folder: {current_run_type_folder})")

    # 2. Setup Paths
    RUN_HOME = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{BASE_NAME}"
    PBT_HOME = path.join(RUN_HOME, f"pbt_runs/{current_run_type_folder}/{RUN_DIR}/")
    lfads_dataset_dir = path.join(RUN_HOME, f"pbt_runs/{current_run_type_folder}/lfads_input")
    model_dir = path.join(PBT_HOME, "pbt_run/best_model")

    # 3. Load Posterior Samples
    ps_filename = "posterior_samples.h5"
    try:
        sampling_output = load_posterior_averages(
            model_dir, merge_tv=True, ps_filename=ps_filename
        )
    except Exception as e:
        logger.warning(f"Skipping {BASE_NAME}: Could not load posterior samples from {model_dir}")
        logger.warning(f"Error Details: {e}")
        continue

    # 4. Process Each Session
    for session_id in session_ids:
        ds_name = f"{ds_base_name}_{session_id}"
        
        # This path is used for LOADING the base data (accumulated spikes)
        final_pkl_path = path.join(nwb_cache_dir, f"nlb_{ds_base_name}_{session_id}.pkl")

        # Path to the step_info pickle file
        step_info_pkl_path = path.join(nwb_cache_dir, f"{ds_name}_step_info.pkl")

    # --- SMART LOADING ---
    if os.path.exists(step_info_pkl_path):
        logger.info(f"Loading STEP INFO dataset: {step_info_pkl_path}")
        with open(step_info_pkl_path, "rb") as rfile:
            dataset = pickle.load(rfile)
    elif os.path.exists(final_pkl_path):
        logger.info(f"Loading ACCUMULATED dataset: {final_pkl_path}")
        with open(final_pkl_path, "rb") as rfile:
            dataset = pickle.load(rfile)
    elif use_cached:
        initial_cache_path = path.join(nwb_cache_dir, "post_pcr", ds_name + "_post_pcr.pkl")
        logger.info(f"Loading INITIAL cache: {initial_cache_path}")
        with open(initial_cache_path, "rb") as rfile:
            dataset = pickle.load(rfile)
    else:
        raw_path = path.join(ds_base_dir, ds_name + ".nwb")
        logger.info("Loading from NWB")
        dataset = NWBDataset(raw_path)

    #     # --- SMART LOADING ---
    #     if os.path.exists(final_pkl_path):
    #         logger.info(f"Loading ACCUMULATED dataset: {final_pkl_path}")
    #         with open(final_pkl_path, "rb") as rfile:
    #             dataset = pickle.load(rfile)
    #     elif use_cached:
    #         initial_cache_path = path.join(nwb_cache_dir, "post_pcr", ds_name + "_post_pcr.pkl")
    #         logger.info(f"Loading INITIAL cache: {initial_cache_path}")
    #         with open(initial_cache_path, "rb") as rfile:
    #             dataset = pickle.load(rfile)
    #     else:
    #         raw_path = path.join(ds_base_dir, ds_name + ".nwb")
    #         logger.info("Loading from NWB")
    #         dataset = NWBDataset(raw_path)

        # 5. Load YAML Config
        yaml_name = f"cfg_{ds_name}*.yaml"
        try:
            cfg_yaml_filepath = glob.glob(os.path.join(lfads_dataset_dir, yaml_name))[0]
            with open(cfg_yaml_filepath, "r") as yamlfile:
                cfg_node = yaml.load(yamlfile, Loader=yaml.FullLoader)
        except IndexError:
             logger.warning(f"Config not found in {lfads_dataset_dir}. Skipping.")
             continue
        
        BIN_SIZE = cfg_node[0]["DATASET"]["BIN_SIZE"]
        DATASET_NAME = cfg_node[0]["DATASET"]["NAME"]
        ARRAY_SELECT_RAW = cfg_node[0]["DATASET"]["ARRAY_SELECT"] 
        DATA_FIELDNAME = cfg_node[1]["CHOP_PARAMETERS"]["DATA_FIELDNAME"]

        # --- FORCE "ALL" FOR EMG ---
        if IS_EMG:
            ARRAY_SELECT = "ALL"
            logger.info("-> Force-setting ARRAY_SELECT to 'ALL' for EMG run.")
        else:
            ARRAY_SELECT = ARRAY_SELECT_RAW

        # 6. Load Interface
        if IS_EMG:
            interface_name = f"{ds_name}_*_emg_10_interface.pkl"
        else:
            interface_name = f"{ds_name}_*_spikes_10_interface.pkl"

        try:
            interface_filepath = glob.glob(os.path.join(lfads_dataset_dir, interface_name))[0]
            with open(interface_filepath, "rb") as rfile:
                interface = pickle.load(rfile)
        except IndexError:
            logger.warning(f"Interface not found for {ds_name}. Skipping session.")
            continue

        # 7. Prepare Data Dictionary
        h5_name = f"*{ds_name}*.h5"
        try:
            lfads_ds_filepath = glob.glob(os.path.join(lfads_dataset_dir, h5_name))[0]
            lfads_ds_filename = lfads_ds_filepath.split("/")[-1] 
            lfads_dataset_name = lfads_ds_filename.replace("lfads_", "")
        except IndexError:
             logger.warning(f"H5 dataset not found for {ds_name}. Skipping.")
             continue

        # Define Load Names
        base_spikes_load_names = ["lfads_rates", "lfads_factors", "lfads_gen_inputs"]
        base_emg_load_names = ["deEMG_mean", "deEMG_var", "deEMG_factors", "deEMG_gen_inputs"]

        emg_load_names = [ms_prefix + n for n in base_emg_load_names]
        spikes_load_names = [ms_prefix + n for n in base_spikes_load_names]
        
        data_dict = {}
        data_dict["factors"] = sampling_output[lfads_dataset_name].factors
        data_dict["gen_inputs"] = sampling_output[lfads_dataset_name].gen_inputs

        # --- LOGIC BRANCH ---
        if not IS_EMG:
            # SPIKES
            interface.merge_fields_map = {
                "output_params": spikes_load_names[0],
                "factors": spikes_load_names[1],
                "gen_inputs": spikes_load_names[2],
            }
            data_dict["output_params"] = np.squeeze(sampling_output[lfads_dataset_name].output_params)
            load_names = spikes_load_names
            match_col_names = [True, False, False]
        else:
            # EMG
            interface.merge_fields_map = {
                "output_par_1": emg_load_names[0],
                "output_par_2": emg_load_names[1],
                "factors": emg_load_names[2],
                "gen_inputs": emg_load_names[3],
            }
            data_dict["output_par_1"] = np.squeeze(sampling_output[lfads_dataset_name].output_params[:, :, :, 0])
            data_dict["output_par_2"] = np.squeeze(sampling_output[lfads_dataset_name].output_params[:, :, :, 1])
            load_names = emg_load_names
            match_col_names = [True, True, False, False]

        # 8. Merge
        cts_df = interface.merge(data_dict, smooth_pwr=1)

        # 9. Determine Side Suffix
        all_locations = dataset.unit_info['location'].values.astype(str)
        
        if ARRAY_SELECT == "L":
            candidate_indices = np.where([("L side" in loc) for loc in all_locations])[0]
            side_suffix = "_L"
        elif ARRAY_SELECT == "R":
            candidate_indices = np.where([("R side" in loc) for loc in all_locations])[0]
            side_suffix = "_R"
        elif ARRAY_SELECT == "ALL":
            candidate_indices = np.where([("L side" in loc or "R side" in loc) for loc in all_locations])[0]
            side_suffix = "_ALL"

        # 10. Sort Candidates
        raw_spike_counts = []
        for idx in candidate_indices:
            if idx in dataset.data[DATA_FIELDNAME].columns:
                raw_spike_counts.append(dataset.data[DATA_FIELDNAME][idx].sum())
            else:
                raw_spike_counts.append(0)
        sorted_pairs = sorted(zip(candidate_indices, raw_spike_counts), key=lambda x: x[1], reverse=True)
        sorted_candidates = np.array([p[0] for p in sorted_pairs])

        # 11. Add Data
        MO_FIELDS = load_names
        MATCH_FIELDNAMES = match_col_names

        for MO_FIELD, MATCH_FIELDNAME in zip(MO_FIELDS, MATCH_FIELDNAMES):
            data_to_add = cts_df[MO_FIELD].values
            save_name = MO_FIELD + name_mod_suffix + side_suffix 

            chan_names = None
            if MATCH_FIELDNAME and len(sorted_candidates) > 0:
                num_data_cols = data_to_add.shape[1]
                if num_data_cols < len(sorted_candidates):
                    kept_indices = sorted_candidates[:num_data_cols]
                    chan_names = np.sort(kept_indices)
                elif num_data_cols >= len(sorted_candidates):
                    chan_names = np.sort(sorted_candidates)
                    if num_data_cols > len(chan_names):
                         data_to_add = data_to_add[:, :len(chan_names)]
            
            if save_name in dataset.data.columns.levels[0]:
                dataset.data = dataset.data.drop([save_name], level=0, axis=1) 
            
            logger.info(f"Adding: {save_name} | Shape: {data_to_add.shape}")
            dataset.add_continuous_data(data_to_add, save_name, chan_names=chan_names)

        # 12. SAVE
        dataset.data = dataset.data.loc[:, ~dataset.data.columns.duplicated()]
        
        if IS_EMG:
            # Save EMG to separate file
            save_path = path.join(nwb_cache_dir, f"nlb_{DATASET_NAME}_{session_id}_emg.pkl")
            logger.info(f"Saving Separate EMG File: {save_path}")
        else:
            # Update main Spikes file
            save_path = final_pkl_path 
            logger.info(f"Updating Main Spikes File: {save_path}")

        with open(save_path, "wb") as rfile:
            pickle.dump(dataset, rfile, protocol=4)

print("All runs processed successfully.")
# %%