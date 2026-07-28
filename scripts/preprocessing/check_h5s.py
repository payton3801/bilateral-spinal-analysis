# %%
import h5py
import numpy as np
import glob
import os

data_dir = "/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_ALL/pbt_runs/spikes/lfads_input"
files = glob.glob(os.path.join(data_dir, "*.h5"))

print(f"Verifying {len(files)} files...")
bad_files = 0

for fpath in files:
    if "pcr_alignment" in fpath: continue
    
    with h5py.File(fpath, 'r') as f:
        for key in ['train_data', 'valid_data']:
            if key in f:
                data = f[key][()]
                # Check for exact zeros
                if np.any(data == 0):
                    print(f"❌ FAIL: {os.path.basename(fpath)} still has ZEROS!")
                    bad_files += 1
                # Check for NaNs
                if np.any(np.isnan(data)):
                    print(f"❌ FAIL: {os.path.basename(fpath)} has NaNs!")
                    bad_files += 1

if bad_files == 0:
    print(" PASS: Data is clean (No zeros, No NaNs).")
    print("If loss is still NaN, the issue is your PCR Initialization weights (Exploding Gradients).")
else:
    print(f" FOUND {bad_files} BAD FILES. The patch did not work.")

# %%
#replacing 0s with tiny values
import h5py
import numpy as np
import glob
import os

# === PATH TO YOUR LFADS INPUT DATA ===
# Verify this path matches where your .h5 files are located
data_dir = "/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_ALL/pbt_runs/spikes/lfads_input"

files = glob.glob(os.path.join(data_dir, "lfads_*.h5"))
print(f"Scanning {len(files)} files for zeros...")

epsilon = 1e-4

for fpath in files:
    # Skip the alignment matrix file
    if "pcr_alignment" in fpath: 
        continue
    
    with h5py.File(fpath, 'r+') as f: # 'r+' allows reading and writing
        for key in ['train_data', 'valid_data']:
            if key in f:
                data = f[key][()]
                
                # Check for exact zeros
                if np.any(data == 0):
                    print(f"  Patching zeros in {os.path.basename(fpath)} [{key}]")
                    
                    # Replace 0.0 with epsilon (preserve other values)
                    data = np.maximum(data, epsilon)
                    
                    # Overwrite the dataset
                    del f[key]
                    f.create_dataset(key, data=data)

print("Done. You can now run LFADS with RECON_TYPE: 'Gamma'.")
# %%
