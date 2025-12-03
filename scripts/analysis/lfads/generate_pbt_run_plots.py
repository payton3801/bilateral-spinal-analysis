# %%
from tune_tf2.pbt.utils import plot_pbt_log, plot_pbt_hps, read_pbt_fitlog, read_pbt_hps
import matplotlib.pyplot as plt
from os import path

# the name of this PBT run (run will be stored at {PBT_HOME}/{PBT_NAME})
# PBT_HOME = '/snel/share/share/derived/interaction/rtg/runs/emg_run_001/'
# PBT_HOME = '/snel/share/share/derived/Tresch/nwb_lfads/runs/run_007'
# cat_name = "black"
cat_name = ""
#cat_name = "woody"
# cat_name = "munch"
# cat_name = "bev"
#run_type = "spikes"
run_type = "spikes"
#base_name = "binsize_10ms_all_sess"
# base_name = "binsize_4ms_ALL"
#base_name = "binsize_4ms_pcr_high_reg_2_ALL"
base_name = "binsize_10ms_pcr_L"
run_dir = "run_pcr_freeze"
# trial_prefix = "trial_full"
# PBT_HOME = f"/snel/share/share/derived/interaction/rtg/nwb_lfads/runs/init_modeling/{trial_prefix}/{run_type}/run_000/"
#PBT_HOME = f"/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/{base_name}/pbt_runs/{run_type}/{run_dir}/"
PBT_HOME =  "/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_L/pbt_runs/spikes/"
# PBT_HOME = f"/snel/share/share/derived/auyong/nwb_lfads/runs/robust_test/{run_type}/run_008/"

RUN_NAME = "run_001/pbt_run"  # the name of the PBT run
PBT_DIR = path.join(PBT_HOME, RUN_NAME)

fitlog = read_pbt_fitlog(PBT_DIR)
hps = read_pbt_hps(PBT_DIR)
plot_pbt_log(PBT_DIR, "recon_heldin")
plot_pbt_log(PBT_DIR, "val_recon_heldin")
plot_pbt_hps(PBT_DIR, "TRAIN.KL.CO_WEIGHT")
plot_pbt_hps(PBT_DIR, "TRAIN.KL.IC_WEIGHT")
plot_pbt_hps(PBT_DIR, "TRAIN.L2.GEN_SCALE")
plot_pbt_hps(PBT_DIR, "TRAIN.L2.CON_SCALE")
plot_pbt_hps(PBT_DIR, "MODEL.CD_RATE")
plot_pbt_hps(PBT_DIR, "MODEL.DROPOUT_RATE")
plot_pbt_hps(PBT_DIR, "TRAIN.LR.INIT")
# plot_pbt_hps(PBT_DIR, 'TRAIN.L2.EXT_INPUT_READIN_SCALE')
plt.show()
# import pdb; pdb.set_trace()
# %%
