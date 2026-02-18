import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'runs/classify/DAPWH_yolo26n-cls/weights/best.pt'
TEST_DIR = Path('data/processed/DAPWH_Final_Split/test')
OUTPUT_BASE = Path('reports/YOLO_CAM')


# 1. Load the SOTA Model
model = YOLO(MODEL_PATH)

def run_full_cam():
    # Support for all requested formats (including high-res TIF/PNG)
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    
    # Gather all images from the test set
    image_files = [p for p in TEST_DIR.rglob('*') if p.suffix.lower() in extensions]
    total_imgs = len(image_files)
    
    print(f"Starting full analysis of {total_imgs} images on RTX 4090...")

    for idx, img_path in enumerate(image_files):
        try:
            # Use PIL for TIF compatibility, then convert for OpenCV/YOLO
            img_pil = Image.open(img_path).convert('RGB')
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Generate visualization
            # name defines the subfolder for this specific image's feature maps
            results = model.predict(
                source=img_bgr,
                imgsz=512,
                visualize=True, 
                name=f"CAM_{img_path.stem}",
                project=OUTPUT_BASE,
                exist_ok=True
            )

            if idx % 50 == 0:
                print(f"Processed {idx}/{total_imgs} images...")
                torch.cuda.empty_cache() # Prevent VRAM fragmentation

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

if __name__ == "__main__":
    run_full_cam()
    print(f" Full analysis complete. Results saved in: {OUTPUT_BASE}")