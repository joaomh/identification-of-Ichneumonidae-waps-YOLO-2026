import os
import random
import shutil
from PIL import Image

def prepare_dataset(source_dir, output_dir, img_size=(512, 512)):
    # Create directory structure for YOLO
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # List all images
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.tif'))]
    random.shuffle(images)

    # Calculate split indices (60% Train, 20% Val, 20% Test)
    total = len(images)
    train_end = int(total * 0.6)
    val_end = train_end + int(total * 0.2)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    print(f"Total images: {total}")
    for split, files in splits.items():
        print(f"Processing {split} set ({len(files)} images)...")
        for f in files:
            # Rescale and save image
            with Image.open(os.path.join(source_dir, f)) as img:
                img_resized = img.resize(img_size, Image.LANCZOS)
                img_resized.save(os.path.join(output_dir, split, 'images', f))
            
            # Note: Labels (.txt) should be moved to corresponding 'labels' folder 
            # if they already exist in the source directory.

if __name__ == "__main__":
    SOURCE = "data/raw"
    OUTPUT = "data/processed"
    prepare_dataset(SOURCE, OUTPUT)