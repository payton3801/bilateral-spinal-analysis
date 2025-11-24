#%%
import sys
import os
import h5py
import _pickle as pickle
import logging
import yaml
import pandas as pd
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# to load from pickled rds
lfads_save_dir = '/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets'
cache_dir = '/snel/share/share/data/Tresch_gaitEMG/data/NWB/'
dataset_name = 'J10_s20_i0'
cached_path = os.path.join(cache_dir, dataset_name+'.nwb')
time_limit = None # seconds
# to load from tmp path of pickled rds (helpful if rds has already been preprocessed)
#cached_path = '/tmp/lwimala/jango_2016_w_lfads_rates.pkl'
lfads_dataset_cfg = [
    {
        'DATASET': {
            'NAME': dataset_name,
            'CONDITION_SEP_FIELD': None, # continuous
            'ALIGN_LIMS': None,
            'TIME_LIMIT': time_limit, # seconds
            'BIN_SIZE': 2,
            'EXCLUDE_TRIALS': [],
            'EXCLUDE_CONDITIONS': [],
            'EXCLUDE_CHANNELS': []
        }
    },
    {
        'CHOP_PARAMETERS': {
            'TYPE': 'emg',
            'DATA_FIELDNAME': 'emg',
            'USE_EXT_INPUT': False,
            'EXT_INPUT_FIELDNAME': '',
            'WINDOW': 200, #ms
            'OVERLAP': 50, #ms
            'MAX_OFFSET': 0,
            'RANDOM_SEED': 0,
            'CHOP_MARGINS': 0
        }
    }
]

if lfads_dataset_cfg[1]['CHOP_PARAMETERS']['TYPE'] == 'emg':
    emg_chop_cfg = {
        'CLIP_QUANTILE': [ 0.999, 0.999, 0.999,
                           0.999, 0.999, 0.999,
                           0.99,  0.999, 0.999, # SM has lower cutoff
                           0.999, 0.999, 0.999],
        'SCALE_QUANTILE': 0.95,
        'SCALE_FACTOR': 1,
        'LOG_TRANSFORM': False
    }
    logger.info(f"Updating chop config with EMG scaling parameters")
    lfads_dataset_cfg[1]['CHOP_PARAMETERS'].update(emg_chop_cfg)


logger.info('Loading from NWB')

from snel_toolkit.datasets.nwb import NWBDataset
dataset = NWBDataset(cached_path)


ld_cfg = lfads_dataset_cfg[0]['DATASET']

if ld_cfg['BIN_SIZE'] != dataset.bin_width:
    logger.info(f"Resampling  dataset to bin width (s): {ld_cfg['BIN_SIZE']}")
    dataset.resample(ld_cfg['BIN_SIZE'])
'''
# get trial info from dataset
ti = dataset.trial_info

excluded_kin_trial_ids = ld_cfg['EXCLUDE_TRIALS']
good_kin = ti.trial_id != -1 # all true
for trial_id in exclude_kin_trial_ids:
    good_kin[ti.trial_id==trial_id] = False
good_cycle = nwb_s10.trial_info.good_cycle == 1

ignore_trials = (~good_cycle | ~good_kin)


# make trial data
td = dataset.make_trial_data(
    align_field='move_onset',
    align_range=(ld_cfg['ALIGN_LIMS'][0],ld_cfg['ALIGN_LIMS'][1]), # ms
    allow_overlap=True,
    ignored_trials=~keep_trials
)
'''

if time_limit is not None:
    logger.info(f'Limiting dataframe for modeling to {time_limit} seconds')
    keep_inds = (dataset.data.index < pd.to_timedelta(time_limit, unit='s'))
    chop_df = dataset.data.iloc[keep_inds,:]
else:
    chop_df = dataset.data

pkl_dir = os.path.join(lfads_save_dir,'pkls')
if os.path.exists(lfads_save_dir) is not True:
    os.makedirs(lfads_save_dir)
if os.path.exists(pkl_dir) is not True:
    os.makedirs(pkl_dir)


# initialize chop interface
chop_cfg = lfads_dataset_cfg[1]['CHOP_PARAMETERS']
# setup initial chop fields map (defines which fields will be chopped for lfads)
chop_fields_map = {chop_cfg['DATA_FIELDNAME']: 'data'}
# if we are using external inputs, add this field to chop map
if chop_cfg['USE_EXT_INPUT']:
    logger.info(f"Setting up lfads dataset with external inputs from {chop_cfg['EXT_INPUT_FIELDNAME']}")
    chop_fields_map[chop_cfg['EXT_INPUT_FIELDNAME']] = 'ext_input'

from snel_toolkit.interfaces import LFADSInterface

interface = LFADSInterface(
    window=chop_cfg['WINDOW'],
    overlap=chop_cfg['OVERLAP'],
    max_offset=chop_cfg['MAX_OFFSET'],
    chop_margins=chop_cfg['CHOP_MARGINS'],
    random_seed=chop_cfg['RANDOM_SEED'],
    chop_fields_map=chop_fields_map 
)  
# save dataset for each session

ds_name = 'lfads_' + ld_cfg['NAME'] + '_' + chop_cfg['TYPE'] + '_' + str(ld_cfg['BIN_SIZE']) + '.h5'
yaml_name = 'cfg_' + ld_cfg['NAME'] + '_' + chop_cfg['TYPE'] + '_' + str(ld_cfg['BIN_SIZE']) + '.yaml'

DATA_FILE = os.path.join(lfads_save_dir, ds_name) #add something that pickles data file at the path data file
YAML_FILE = os.path.join(lfads_save_dir, yaml_name)

interface.chop_and_save(chop_df, DATA_FILE, overwrite=True)
INTERFACE_FILE = os.path.join(pkl_dir, ld_cfg['NAME'] + '_' + chop_cfg['TYPE'] + '_' + str(ld_cfg['BIN_SIZE']) +'_interface.pkl')
with open(YAML_FILE,'w') as yamlfile:
    data1 = yaml.dump(lfads_dataset_cfg, yamlfile)
    yamlfile.close()
with open(INTERFACE_FILE, 'wb') as rfile:
    logger.info('Interface {} saved to pickle.'.format(INTERFACE_FILE))
    pickle.dump(interface,rfile)

# %%

# Save the YAML file to the current working directory
current_folder = "/home/pbechef/emg_data_analysis/Data_Files/"  
YAML_FILE = os.path.join(current_folder, yaml_name)

# Write the YAML file
with open(YAML_FILE, 'w') as yamlfile:
    data1 = yaml.dump(lfads_dataset_cfg, yamlfile)
    yamlfile.close()

print(f"YAML file saved to: {YAML_FILE}")

# save the data file as a pickle file in current working directory
ds_name_pkl = 'lfads_' + ld_cfg['NAME'] + '_' + chop_cfg['TYPE'] + '_' + str(ld_cfg['BIN_SIZE']) + '.pkl'

current_folder = "/home/pbechef/emg_data_analysis/Data_Files/"
DATA_FILE = os.path.join(current_folder, ds_name_pkl)

with open(DATA_FILE, 'wb') as pkl_file:
    pickle.dump(dataset, pkl_file)

print(f"Data file pickled and saved to: {DATA_FILE}")

# %%
