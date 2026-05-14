import os
from PIL import Image
import zipfile

# --- CONFIGURATION ---
# CHANGE THIS to the actual folder where your 63k images are extracted
SOURCE_IMAGES = r'C:\Users\vishw\Downloads\archive\data' 
TARGET_FOLDER = 'anime_subset_256'
ZIP_NAME = 'datasets/anime_subset.zip'
# ---------------------

os.makedirs(TARGET_FOLDER, exist_ok=True)
os.makedirs('datasets', exist_ok=True)

print("Starting image processing...")
files = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
subset = files[:1000] # Grabbing 1,000 images for high quality few-shot

for i, filename in enumerate(subset):
    try:
        img = Image.open(os.path.join(SOURCE_IMAGES, filename)).convert('RGB')
        # Force 256x256 for FFHQ compatibility
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(os.path.join(TARGET_FOLDER, f"img_{i:04d}.png"))
        if i % 100 == 0:
            print(f"Processed {i}/1000...")
    except Exception as e:
        print(f"Skipping {filename} due to error.")

print("Creating ZIP file for training...")
with zipfile.ZipFile(ZIP_NAME, 'w') as z:
    for filename in os.listdir(TARGET_FOLDER):
        z.write(os.path.join(TARGET_FOLDER, filename), filename)

print(f"\nSUCCESS! Your dataset is ready at: {os.path.abspath(ZIP_NAME)}")