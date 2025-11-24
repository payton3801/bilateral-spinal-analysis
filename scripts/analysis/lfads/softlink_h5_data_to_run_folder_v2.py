# %%
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
ARRAY_SELECT = "L"
base_name = F"binsize_10ms_pcr_{ARRAY_SELECT}"
RUN_HOME = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{base_name}/"
RUN_TYPE = "spikes"
#RUN_TYPE = "emg"
#SIDE = "R"
ARRAY_SELECT = "L"
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
DS_WILDCARD = f"*{ARRAY_SELECT}*{RUN_TYPE}*"
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
subprocess.run(shlex.split(bash_cmd_2))
# %%
