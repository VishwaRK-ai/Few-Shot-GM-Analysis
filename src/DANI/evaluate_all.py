import os
import glob
import subprocess

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --- CONFIGURATION ---
DANI_DIR = os.path.join(BASE_DIR, "src", "DANI")
RUN_DIR = os.path.join(BASE_DIR, "src", "DANI", "training-runs", "00004-stylegan2-panda-gpus1-batch8-d_pos-first-noise_sd-0.5-target0.45-ada_kimg100-brand_new_run_numbered")
DATASET_PATH = os.path.join(BASE_DIR, "src", "FakeCLR", "data", "panda.zip")
PYTHON_EXEC = os.path.join(BASE_DIR, "src", "DANI", "dani_env", "Scripts", "python.exe")

# All three metrics!
METRICS = "kid50k_full,fid50k_full,is50k" 

# --- FIND FILES & APPLY SPEED HACK ---
search_path = os.path.join(RUN_DIR, "network-snapshot-*.pkl")
all_pkl_files = sorted(glob.glob(search_path))

if len(all_pkl_files) == 0:
    print(f"ERROR: No files found in {search_path}")
    exit()

STEP_SIZE = 1
pkl_files_to_test = all_pkl_files[::STEP_SIZE]

# Ensure the very last snapshot is ALWAYS evaluated so you know your final score
if all_pkl_files[-1] not in pkl_files_to_test:
    pkl_files_to_test.append(all_pkl_files[-1])

print("\n" + "="*70)
print("STARTING FAST METRIC EVALUATION")
print("="*70)
print(f"Total snapshots found in folder: {len(all_pkl_files)}")
print(f"Snapshots evaluating today (to save time): {len(pkl_files_to_test)}")
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
    
    # By removing the PIPE and capturing logic, the child process 
    # writes directly to your terminal. Progress bars will update instantly.
    result = subprocess.run(cmd, cwd=DANI_DIR)
    
    if result.returncode == 0:
        print(f"\n[✓] Successfully evaluated {snapshot_name}")
        successful_evals += 1
    else:
        print(f"\n[✗] Failed to evaluate {snapshot_name}")
        failed_evals += 1

print("\n" + "="*70)
print(f"EVALUATION COMPLETE - Success: {successful_evals} | Failed: {failed_evals}")
print("="*70)