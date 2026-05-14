import os
import torch
import numpy as np
import dnnlib
import legacy
import PIL.Image

# 1. Force CPU and Float32
device = torch.device('cpu')
network_pkl = "training-runs\\00009-obama-mirror-paper256-resumecustom\\network-snapshot-000004.pkl"
outdir = "fid_temp_fake"
os.makedirs(outdir, exist_ok=True)

print(f"Loading weights onto CPU and converting to Float32...")

with dnnlib.util.open_url(network_pkl) as f:
    data = legacy.load_network_pkl(f)
    G = data['G_ema'].to(device).float() # .float() fixes the 'Half' error

# 2. Generate 100 images
print("Generating 100 images on CPU. This will take a moment...")
G.eval()
with torch.no_grad():
    for seed in range(100):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
        c = torch.zeros([1, G.c_dim], device=device).float()
        
        # Calling forward specifically
        img = G(z, c, truncation_psi=1.0, noise_mode='const')
        
        # Standard conversion to Image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        
        if seed % 10 == 0:
            print(f"Progress: {seed}/100")

print(f"Done! 100 images are in {outdir}. Run your FID command now.")