import os
import sys
import glob
import csv
import torch
import itertools
import numpy as np
from pytorch_msssim import ms_ssim

# --- 1. CONFIGURATION ---
DANI_DIR = os.path.join(BASE_DIR, "src", "DANI")
RUN_DIR = os.path.join(BASE_DIR, "src", "DANI", "training-runs", "00004-stylegan2-panda-gpus1-batch8-d_pos-first-noise_sd-0.5-target0.45-ada_kimg100-brand_new_run_numbered")
OUTPUT_CSV = os.path.join(RUN_DIR, "custom_msssim_metrics.csv")

NUM_IMAGES = 20 

if DANI_DIR not in sys.path:
    sys.path.append(DANI_DIR)

import dnnlib
import legacy
from torch_utils.ops import bias_act, upfirdn2d, conv2d_gradfix, grid_sample_gradfix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


bias_act.enabled = False
upfirdn2d.enabled = False
conv2d_gradfix.enabled = False
grid_sample_gradfix.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. FIND SNAPSHOTS ---
search_path = os.path.join(RUN_DIR, "network-snapshot-*.pkl")
pkl_files_to_test = sorted(glob.glob(search_path))

if not pkl_files_to_test:
    print(f"ERROR: No files found in {search_path}")
    sys.exit()

print(f"Found {len(pkl_files_to_test)} snapshots. Beginning MS-SSIM batch evaluation...\n")

# --- 3. BATCH EVALUATION LOOP ---
with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['snapshot_pkl', 'kimg', 'msssim_avg'])

    for idx, pkl_file in enumerate(pkl_files_to_test, 1):
        snapshot_name = os.path.basename(pkl_file)
        kimg = int(snapshot_name.split('-')[-1].split('.')[0])
        
        print(f"[{idx}/{len(pkl_files_to_test)}] Evaluating: {snapshot_name}")
        
        with dnnlib.util.open_url(pkl_file) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
        torch.manual_seed(42)
        z = torch.randn([NUM_IMAGES, G.z_dim]).to(device)
        c = None
        
        with torch.no_grad():
            img_tensors = G(z, c, truncation_psi=0.7, noise_mode='const')
            # Normalize StyleGAN output from [-1, 1] to [0, 1] for MS-SSIM
            img_tensors = (img_tensors + 1) / 2.0
            
        distances = []
        pairs = list(itertools.combinations(range(NUM_IMAGES), 2))
        
        with torch.no_grad():
            for i, j in pairs:
                img_a = img_tensors[i].unsqueeze(0)
                img_b = img_tensors[j].unsqueeze(0)
                sim = ms_ssim(img_a, img_b, data_range=1.0, size_average=True)
                distances.append(sim.item())
                
        average_msssim = np.mean(distances)
        print(f"    ↳ MS-SSIM Score: {average_msssim:.4f} (Lower = More Diverse)")
        
        csv_writer.writerow([snapshot_name, kimg, average_msssim])
        
        del G
        del img_tensors
        torch.cuda.empty_cache()

print("\n" + "="*70)
print(f"EVALUATION COMPLETE - Results saved to:\n{OUTPUT_CSV}")
print("="*70)