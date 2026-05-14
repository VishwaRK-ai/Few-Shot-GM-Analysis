import urllib.request
import os

print("Fetching updated PyTorch 2.x custom ops from NVIDIA...")

# URLs to the patched files from the newer StyleGAN3 repository
sg3_base = "https://raw.githubusercontent.com/NVlabs/stylegan3/main/torch_utils/ops/"
files = ["grid_sample_gradfix.py", "conv2d_gradfix.py"]

for file in files:
    url = sg3_base + file
    filepath = os.path.join("torch_utils", "ops", file)
    
    # Download and overwrite
    urllib.request.urlretrieve(url, filepath)
    print(f"✅ Replaced: {filepath}")

print("Done. You are clear to launch training.")