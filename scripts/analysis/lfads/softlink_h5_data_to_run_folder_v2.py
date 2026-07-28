# %%
###################
# putting everything we need into lfads_input

## putting datasets in a folder for models to find
import subprocess
import shlex
import os
import glob
import sys
import logging
import yaml


CAT_NAME = "cat03"

# -- setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#%%

# base_name = "binsize_10ms_all_sess"
ARRAY_SELECT = "ALL"
base_name = F"binsize_10ms_pcr_{ARRAY_SELECT}"
RUN_HOME = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{base_name}/"
RUN_TYPE = "emg"
#RUN_TYPE = "spikes"
#SIDE = "R"
PBT_HOME = os.path.join(RUN_HOME, f"pbt_runs/{RUN_TYPE}")
DATASET_DIR = os.path.join(RUN_HOME, "datasets")


align_suffix = ""
#align_suffix = "_low_reg"

PCR_FILE = os.path.join(
    RUN_HOME, f"alignment_matrices/{RUN_TYPE}/pcr_alignment{align_suffix}.h5"
)
INPUT_DIR = os.path.join(PBT_HOME, "lfads_input")
#INPUT_DIR = os.path.join(PBT_HOME, f"lfads_input{align_suffix}")

if not os.path.isdir(INPUT_DIR):
    logger.info(f"Creating {INPUT_DIR}")
    os.makedirs(INPUT_DIR)
DS_WILDCARD = f"*{RUN_TYPE}*"
#DS_WILDCARD = f"*{ARRAY_SELECT}*{RUN_TYPE}*"
#DS_WILDCARD_2 = f"*{SIDE}*"
PKL_DS_WILDCARD = f"pkls/*{ARRAY_SELECT}*{RUN_TYPE}*"
DS_PATH = os.path.join(DATASET_DIR, DS_WILDCARD)
PKL_DS_PATH = os.path.join(DATASET_DIR, PKL_DS_WILDCARD)
# input_path = os.path.join(input_dir, ds_filename)
# bash_cmd_0 = f"unlink {input_path}"
ds_files = glob.glob(DS_PATH, recursive=False)
pkl_ds_files = glob.glob(PKL_DS_PATH)

# %%
for ds_file in ds_files:
    bash_cmd_1 = f"ln -s {ds_file} {INPUT_DIR}/"
    logger.info(f"Running {bash_cmd_1}")
    subprocess.run(shlex.split(bash_cmd_1))
for ds_file in pkl_ds_files:
    bash_cmd_1 = f"ln -s {ds_file} {INPUT_DIR}/"
    logger.info(f"Running {bash_cmd_1}")
    subprocess.run(shlex.split(bash_cmd_1))

bash_cmd_2 = f"ln -s {PCR_FILE} {INPUT_DIR}/pcr_alignment.h5"
logger.info(f"Running {bash_cmd_2}")
subprocess.run(shlex.split(bash_cmd_2))# %%


# %%%%
#fixed emg code


# -- setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
ARRAY_SELECT = "ALL"
base_name = f"binsize_10ms_pcr_{ARRAY_SELECT}"
RUN_HOME = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{base_name}/"
RUN_TYPE = "emg"
PBT_HOME = os.path.join(RUN_HOME, f"pbt_runs/{RUN_TYPE}")
DATASET_DIR = os.path.join(RUN_HOME, "datasets")
INPUT_DIR = os.path.join(PBT_HOME, "lfads_input")

# --- PATH FIX FOR PKLS ---
# Point directly to the subdirectory where find located your files
PKL_DIR = os.path.join(DATASET_DIR, "pkls") 

if not os.path.isdir(INPUT_DIR):
    logger.info(f"Creating {INPUT_DIR}")
    os.makedirs(INPUT_DIR)

# File patterns
DS_PATH = os.path.join(DATASET_DIR, f"*{RUN_TYPE}*")
PKL_DS_PATH = os.path.join(PKL_DIR, f"*interface.pkl") # Simplest pattern to match your find results

ds_files = glob.glob(DS_PATH)
pkl_ds_files = glob.glob(PKL_DS_PATH)

def create_soft_link(src, dest_dir, link_name=None):
    """Helper to remove existing link and create new one safely."""
    target_name = link_name if link_name else os.path.basename(src)
    link_path = os.path.join(dest_dir, target_name)
    
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path) # Clear old link to prevent 'File exists' error
        
    cmd = f"ln -s {src} {link_path}"
    logger.info(f"Linking: {os.path.basename(src)}")
    subprocess.run(shlex.split(cmd))

# Link .h5 files
for ds_file in ds_files:
    if os.path.isfile(ds_file):
        create_soft_link(ds_file, INPUT_DIR)

# Link .pkl interface files
if not pkl_ds_files:
    logger.error(f"NO PKL FILES FOUND AT: {PKL_DS_PATH}")
else:
    for pkl_file in pkl_ds_files:
        create_soft_link(pkl_file, INPUT_DIR)

# Link PCR alignment file
PCR_FILE = os.path.join(RUN_HOME, f"alignment_matrices/{RUN_TYPE}/pcr_alignment.h5")
if os.path.exists(PCR_FILE):
    create_soft_link(PCR_FILE, INPUT_DIR, link_name="pcr_alignment.h5")
else:
    logger.warning("PCR_FILE not found!")
# %%
