import os
import glob
import subprocess

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


dani_dir = os.path.join(BASE_DIR, "src", "DANI")
run_dir = os.path.join(BASE_DIR, "src", "DANI", "training-runs", "00004-stylegan2-panda-gpus1-batch8-d_pos-first-noise_sd-0.5-target0.45-ada_kimg100-brand_new_run_numbered")
dataset_path = os.path.join(BASE_DIR, "src", "FakeCLR", "data", "panda.zip")
python_exec = os.path.join(BASE_DIR, "src", "DANI", "dani_env", "Scripts", "python.exe")

metrics_to_calc = "kid50k_full" # Just doing KID to keep it fast

search_path = os.path.join(run_dir, "network-snapshot-*.pkl")
all_pkl_files = sorted(glob.glob(search_path))

for pkl in all_pkl_files:
    print(f"\nEvaluating: {os.path.basename(pkl)}")
    cmd = [python_exec, "calc_metrics.py", f"--metrics={metrics_to_calc}", f"--network={pkl}", f"--data={dataset_path}"]
    subprocess.run(cmd, cwd=dani_dir)