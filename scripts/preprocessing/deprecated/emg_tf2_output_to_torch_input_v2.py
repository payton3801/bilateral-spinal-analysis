"""
This script takes the output of setup_lfads_datasets.py and converts it to a format that can be used as input to the LFADS torch model.
run in the data_prep directory with the command: python convert_tf2_output_to_torch_input.py
"""
# i dont think we have to use this - confirm

# %%
import os
import sys
import h5py
import yaml


# load yaml config file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml" # need to change this?
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)
path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg["DATASET"]
chop_cfg = lfads_dataset_cfg["CHOP_PARAMETERS"]

expt_name = ld_cfg["NAME"]
#ARRAY_SELECT = ld_cfg["ARRAY_SELECT"]
TYPE = path_config["TYPE"]
BIN_SIZE = ld_cfg["BIN_SIZE"]
#kin_name = ld_cfg["KINEMATICS_NAME"]

# %%
# -- paths
#base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
#ds_base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
lfads_save_dir = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/'
ds_name = (
    f"lfads_speed_10_incline_" + angle + "h5" #adjust angle and speed here
)
tf2_torch_file = os.path.join(lfads_save_dir, ds_name)
# Open the file
with h5py.File(tf2_torch_file, 'r') as dataset:
    # Print all group names
    train_encod_data = dataset["train_data"][:]
    valid_encod_data = dataset["valid_data"][:]
    train_recon_data = dataset["train_data"][:]
    valid_recon_data = dataset["valid_data"][:]
    train_inds = dataset["train_inds"][:]
    valid_inds = dataset["valid_inds"][:]
    if chop_cfg['USE_EXT_INPUT']:
        train_ext_input = dataset["train_ext_input"][:]
        valid_ext_input = dataset["valid_ext_input"][:]
# %%
torch_dataset_str = "TORCH_" + ds_name

kwargs = dict(dtype='float32', compression='gzip')
with h5py.File(f"../../lfads-torch/datasets/{torch_dataset_str}", 'w') as h5f:

    h5f.create_dataset('train_encod_data', data=train_encod_data, **kwargs)
    h5f.create_dataset('valid_encod_data', data=valid_encod_data, **kwargs)
    h5f.create_dataset('train_recon_data', data=train_recon_data, **kwargs)
    h5f.create_dataset('valid_recon_data', data=valid_recon_data, **kwargs)
    h5f.create_dataset('train_inds', data=train_inds, **kwargs)
    h5f.create_dataset('valid_inds', data=valid_inds, **kwargs)
    if chop_cfg['USE_EXT_INPUT']:
        h5f.create_dataset('train_ext_input', data=train_ext_input, **kwargs)
        h5f.create_dataset('valid_ext_input', data=valid_ext_input, **kwargs)
    print(f"File saved: {torch_dataset_str}")



# %%