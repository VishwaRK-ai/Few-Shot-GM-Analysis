import os
import torch
import numpy as np
import dnnlib
import legacy
import PIL.Image

# 1. Setup paths
outdir = "fid_temp_fake"
os.makedirs(outdir, exist_ok=True)
network_pkl = "training-runs\\00009-obama-mirror-paper256-resumecustom\\network-snapshot-000004.pkl"

print("1/3: Loading model on CPU and forcing Float32 override...")
# 2. Load and aggressively cast to float
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cpu').float()

print("2/3: Generating 100 images (This will take 3-5 minutes)...")
# 3. Generate images with the magic 'force_fp32=True' flag to bypass the 'Half' error
with torch.no_grad():
    for seed in range(100):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to('cpu').float()
        c = torch.zeros([1, G.c_dim], device='cpu').float()
        
        # force_fp32=True completely disables the broken FP16 paths
        img = G(z, c, truncation_psi=1.0, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

print("3/3: Generation complete. Calculating FID Score now...")
# 4. Automatically run the FID calculation
os.system(f"python -m torch_fidelity --fid --input1 {outdir} --input2 datasets/obama.zip")