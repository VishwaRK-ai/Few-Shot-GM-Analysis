import os
import glob
import subprocess

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- CONFIGURATION ---
INSGEN_DIR = os.path.join(BASE_DIR, "src", "insgen")
RUN_DIR = os.path.join(BASE_DIR, "src", "insgen", "training-runs", "00003-panda-mirror-paper256-kimg200-resumecustom")
DATASET_PATH = os.path.join(BASE_DIR, "src", "FakeCLR", "data", "panda.zip")
PYTHON_EXEC = "python" # Assuming you are in the same environment

METRICS = "kid5k_full,fid5k_full,pr5k3_full,is5k" 

# Find all network snapshots in the folder
search_path = os.path.join(RUN_DIR, "network-snapshot-*.pkl")
pkl_files_to_test = sorted(glob.glob(search_path))

print("\n" + "="*70)
print(f"STARTING BATCH 5K EVALUATION (INSGEN) - {len(pkl_files_to_test)} Snapshots")
print("="*70 + "\n")

for idx, pkl_file in enumerate(pkl_files_to_test, 1):
    snapshot_name = os.path.basename(pkl_file)
    print(f"\n[{idx}/{len(pkl_files_to_test)}] Evaluating: {snapshot_name}")
    
    cmd = [
        PYTHON_EXEC, "calc_metrics.py",
        f"--metrics={METRICS}",
        f"--network={pkl_file}",
        f"--data={DATASET_PATH}"
    ]
    
    subprocess.run(cmd, cwd=INSGEN_DIR)