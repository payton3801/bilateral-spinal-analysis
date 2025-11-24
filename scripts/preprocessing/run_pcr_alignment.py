#puts all sessions into a list to reference, also uses proper argv naming

import glob
import subprocess
import shlex

""" Run script to make datasets for LFADS model training"""

cat_name = "cat03"



# base_dir = f"/snel/share/share/data/auyong/nick/{cat_name}/"
base_dir = f"/snel/share/share/derived/auyong/NWB/"
sessions = glob.glob(base_dir + f"{cat_name}*")

session_nums = [sess.split("/")[-1].split("_")[-1].split(".")[0] for sess in sessions]
session_nums = sorted(session_nums)

if cat_name == "cat03":
    # exclude_sessions = [ '001', '002', '003'] # cat03
    #exclude_sessions = ["001", "002", "021", "023", "003"]  # cat03
    exclude_sessions = ["001", "002", "003", "005", "007", "009", "011", "012", "015", "017", "019", "021", "023", "025", "027", "029", "031", "033", "035", "063", "065", "067"]  # cat03 

"""
exclude_sessions = [
    "001",
    "002",
    "021",
    "023",
    "003",  # below rejected based on population state analysis
]  # cat03

    "009",
    "005",
    "007",
    "029",
    "037",
    "047",
    "019",
    "051",
    "063",
    "065",
    "045",
"""


valid_sessions = []
for sess_num in session_nums:
    if sess_num not in exclude_sessions:
        valid_sessions.append(sess_num)

print(f"Found {len(valid_sessions)} valid sessions for alignment.")

sessions_arg = f"[{','.join(valid_sessions)}]"

# 5. Construct and Run the command ONCE
cmd = f"python pcr_alignment_v2.py {cat_name} {sessions_arg}"

try:
    subprocess.run(shlex.split(cmd), check=True)
    print("PCR Alignment complete.")
except subprocess.CalledProcessError as e:
    print(f"Error running PCR alignment: {e}")