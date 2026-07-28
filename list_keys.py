import h5py
import os

# Path to the PCR file we need to inspect
filepath = "/snel/share/share/tmp/scratch/pbechef/bilateral_cat/cat03/nwb_lfads/pcr_alignment.h5"

print(f"--- Listing keys in: {filepath} ---")

try:
    f = h5py.File(filepath, 'r')
    keys = list(f.keys())
    
    if not keys:
        print("The file is empty or does not contain any keys.")
    else:
        print(f"Found {len(keys)} keys:")
        for key in keys:
            print(f"  {key}")
    
    f.close()

except Exception as e:
    print(f"An error occurred: {e}")

print("--- End of list ---")
