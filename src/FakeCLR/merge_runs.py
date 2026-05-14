import os
import shutil
import glob

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Your two training runs
RUN_1 = os.path.join(BASE_DIR, "src", "FakeCLR", "results", "00037-panda-mirror-paper256-kimg200-batch8-resumecustom-freezed2")
RUN_2 = os.path.join(BASE_DIR, "src", "FakeCLR", "results", "00038-panda-mirror-paper256-kimg200-batch8-resumecustom-freezed4")

# The new combined folder
DEST_DIR = os.path.join(BASE_DIR, "src", "FakeCLR", "merged_snapshots")
os.makedirs(DEST_DIR, exist_ok=True)

print("Merging Run 1...")
run1_files = sorted(glob.glob(os.path.join(RUN_1, "network-snapshot-*.pkl")))
for f in run1_files:
    shutil.copy(f, os.path.join(DEST_DIR, os.path.basename(f)))
    print(f"Copied {os.path.basename(f)}")

print("\nMerging Run 2 (Applying +48 Offset)...")
run2_files = sorted(glob.glob(os.path.join(RUN_2, "network-snapshot-*.pkl")))
for f in run2_files:
    num = int(os.path.basename(f).split('-')[-1].split('.')[0])
    
    if num == 0:
        print("Skipped 000000.pkl (Duplicate of 000048.pkl)")
        continue 
        
    new_num = num + 48
    new_name = f"network-snapshot-{new_num:06d}.pkl"
    shutil.copy(f, os.path.join(DEST_DIR, new_name))
    print(f"Renamed {os.path.basename(f)} -> {new_name}")

print(f"\nDone! All files merged safely into: {DEST_DIR}")