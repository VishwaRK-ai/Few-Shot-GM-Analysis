import os
import glob
import subprocess

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- CONFIGURATION ---
ADA_DIR = os.path.join(BASE_DIR, "src", "stylegan2-ADA-pytorch")
RUN_DIR = os.path.join(BASE_DIR, "src", "stylegan2-ADA-pytorch", "training-runs", "00012-panda-mirror-paper256-kimg200-resumecustom")
DATASET_PATH = os.path.join(BASE_DIR, "src", "FakeCLR", "data", "panda.zip")
PYTHON_EXEC = "python" # Automatically uses your activated ada_env2

# Your custom 5k metrics!
METRICS = "kid5k_full,fid5k_full,pr5k3_full,is5k" 

# --- FIND FILES ---
search_path = os.path.join(RUN_DIR, "network-snapshot-*.pkl")
all_pkl_files = sorted(glob.glob(search_path))

if len(all_pkl_files) == 0:
    print(f"ERROR: No files found in {search_path}")
    exit()

# Evaluate EVERY file (Step size 1) because 5k metrics are fast enough to handle it
STEP_SIZE = 1
pkl_files_to_test = all_pkl_files[::STEP_SIZE]

# Ensure the very last snapshot is ALWAYS evaluated
if all_pkl_files[-1] not in pkl_files_to_test:
    pkl_files_to_test.append(all_pkl_files[-1])

print("\n" + "="*70)
print("STARTING LIGHTNING-FAST 5K METRIC EVALUATION (ADA)")
print("="*70)
print(f"Total snapshots found in folder: {len(all_pkl_files)}")
print(f"Snapshots evaluating today: {len(pkl_files_to_test)}")
print(f"Metrics: {METRICS}")
print("="*70 + "\n")

successful_evals = 0
failed_evals = 0

# --- THE DIRECT NATIVE LOOP ---
for idx, pkl_file in enumerate(pkl_files_to_test, 1):
    snapshot_name = os.path.basename(pkl_file)
    print(f"\n\n{'='*70}")
    print(f"[{idx}/{len(pkl_files_to_test)}] Evaluating: {snapshot_name}")
    print(f"{'='*70}\n")
    
    cmd = [
        PYTHON_EXEC,
        "calc_metrics.py",
        f"--metrics={METRICS}",
        f"--network={pkl_file}",
        f"--data={DATASET_PATH}"
    ]
    
    # Run the process directly in the terminal
    result = subprocess.run(cmd, cwd=ADA_DIR)
    
    if result.returncode == 0:
        print(f"\n[✓] Successfully evaluated {snapshot_name}")
        successful_evals += 1
    else:
        print(f"\n[✗] Failed to evaluate {snapshot_name}")
        failed_evals += 1

print("\n" + "="*70)
print(f"EVALUATION COMPLETE - Success: {successful_evals} | Failed: {failed_evals}")
print("="*70)