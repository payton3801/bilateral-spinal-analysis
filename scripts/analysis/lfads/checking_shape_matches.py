
# %%
import h5py

input_file_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_L/pbt_runs/spikes/lfads_input/lfads_cat03_045_L_spikes_10.h5'
posterior_file_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_L/pbt_runs/spikes/run_001/pbt_run/best_model/posterior_samples.h5'

with h5py.File(input_file_path, 'r') as input_f, h5py.File(posterior_file_path, 'r') as posterior_f:
    input_units = input_f['train_data'].shape[-1]
    posterior_units = posterior_f['train_lfads_cat03_045_L_spikes_10.h5/train_factors'].shape[-1]

    if input_units == posterior_units:
        print(f"Consistency check passed: {input_units} units")
    else:
        print(f"Mismatch detected: Input units = {input_units}, Posterior units = {posterior_units}")

# %%
with h5py.File(posterior_file_path, 'r') as f:
    for group_name in f.keys():
        print(f"Group: {group_name}")
        for dataset_name in f[group_name].keys():
            dataset = f[f"{group_name}/{dataset_name}"]
            print(f"  Dataset: {dataset_name}, Shape: {dataset.shape}")
# %%
import h5py

with h5py.File('/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_L/pbt_runs/spikes/run_001/pbt_run/best_model/posterior_samples.h5', 'r') as f:
    # 1. Inspect what is inside the file
    print("Keys available:", list(f.keys()))
    
    # 2. INCORRECT: This is likely what you grabbed (Dim 20)
    # factors = f['factors'][()] 
    
    # 3. CORRECT: Grab the Rates (Should be Dim 30 or 15)
    # Note: might be named 'rates', 'output_dist_params', or 'log_rates'
    if 'rates' in f:
        rates = f['rates'][()]
        print(f"Rates Shape: {rates.shape}") # Should be (Time, 15/30)
        
        # NOW you can slice using your input units
        # real_rates = rates[:, :, :15]
# %%
import h5py

# Path to the posterior samples file
posterior_file_path = '/snel/share/share/tmp/scratch/bilateral_cat/nwb_lfads/runs/binsize_10ms_pcr_L/pbt_runs/spikes/run_001/pbt_run/best_model/posterior_samples.h5'

# Bin size in milliseconds
BIN_SIZE_MS = 10

with h5py.File(posterior_file_path, 'r') as f:
    if 'rates' in f:
        rates = f['rates'][()]
        print(f"Rates Shape: {rates.shape}")  # Shape: (Trials, Time, Channels)
        
        # Calculate recording length for a specific session
        num_time_steps = rates.shape[1]
        recording_length_ms = num_time_steps * BIN_SIZE_MS
        recording_length_sec = recording_length_ms / 1000  # Convert to seconds
        
        print(f"Number of time steps: {num_time_steps}")
        print(f"Recording length: {recording_length_sec} seconds")
# %%
