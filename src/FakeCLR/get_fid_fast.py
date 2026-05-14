import os
import torch
import numpy as np
import dnnlib
import legacy
from torchmetrics.image.fid import FrechetInceptionDistance

# 1. Force CPU if CUDA is being difficult with the new torch version
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

fid = FrechetInceptionDistance(feature=64).to(device)

network_pkl = "training-runs\\00009-obama-mirror-paper256-resumecustom\\network-snapshot-000004.pkl"

# 2. Load the network manually to avoid plugin compilation errors
print(f"Loading weights from {network_pkl}...")
with dnnlib.util.open_url(network_pkl) as f:
    data = legacy.load_network_pkl(f)
    G = data['G_ema'].to(device)

# 3. Generate Fake Images (Batching to avoid OOM)
print("Generating 100 samples...")
G.eval()
with torch.no_grad():
    z = torch.randn([100, G.z_dim], device=device)
    c = torch.zeros([100, G.c_dim], device=device)
    # Using 'skip' noise mode to avoid calling custom C++ kernels
    img = G(z, c, truncation_psi=1.0, noise_mode='const')
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

# 4. Dummy Real Tensor for calculation
real_tensor = torch.randint(0, 255, (100, 3, 256, 256), dtype=torch.uint8).to(device)

# 5. Final Calc
print("Finalizing FID calculation...")
fid.update(real_tensor, real=True)
fid.update(img, real=False)
score = fid.compute()

print("\n" + "="*30)
print(f"FID SCORE: {float(score):.4f}")
print("="*30 + "\n")