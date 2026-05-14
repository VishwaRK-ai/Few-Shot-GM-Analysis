import os
import glob
import subprocess

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- CONFIGURATION ---
FAKECLR_DIR = os.path.join(BASE_DIR, "src", "FakeCLR")
RUN_DIR = os.path.join(BASE_DIR, "src", "FakeCLR", "merged_snapshots")
DATASET_PATH = os.path.join(BASE_DIR, "src", "FakeCLR", "data", "panda.zip")
PYTHON_EXEC = "python" # Assuming you are in the same environment

METRICS = "kid5k_full,fid5k_full,pr5k3_full,is5k" 

search_path = os.path.join(RUN_DIR, "network-snapshot-*.pkl")
pkl_files_to_test = sorted(glob.glob(search_path))

print("\n" + "="*70)
print("STARTING 5K METRIC EVALUATION (FAKECLR)")
print(f"Total snapshots to evaluate: {len(pkl_files_to_test)}")
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
    
    subprocess.run(cmd, cwd=FAKECLR_DIR)