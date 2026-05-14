from datasets import load_dataset
import os
import zipfile

print("Downloading dataset from Hugging Face...")
# This automatically downloads the 100 images
dataset = load_dataset("huggan/few-shot-panda", split="train")

# Ensure the data folder exists
os.makedirs("./data/panda_raw", exist_ok=True)

print("Extracting images...")
image_paths = []
for i, item in enumerate(dataset):
    img = item['image']
    # Save each image as a PNG
    path = f"./data/panda_raw/panda_{i:03d}.png"
    img.save(path)
    image_paths.append(path)

print("Zipping images into panda.zip for FakeCLR...")
with zipfile.ZipFile("./data/panda.zip", "w") as zipf:
    for path in image_paths:
        # Add to zip without the full folder path
        zipf.write(path, os.path.basename(path))

print("✅ Success! panda.zip is ready in your ./data/ folder.")