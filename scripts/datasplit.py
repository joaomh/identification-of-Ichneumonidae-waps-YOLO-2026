import os
import shutil
import splitfolders
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Based on Zenodo record 18501018 structure
BASE_DIR = Path('data/raw/DAPWH') 
# Update VIEWS to match your specific folder names exactly
VIEWS = ['Dorsal_Ventral', 'Frontal', 'Lateral'] 
MERGED_DIR = Path('data/processed/DAPWH_Merged_by_Family')
SPLIT_DIR = Path('data/processed/DAPWH_Final_Split')

def merge_and_split():
    # 1. Merge all views into a single family-based structure
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR) # Clean start to avoid duplicate counting
    MERGED_DIR.mkdir(parents=True)
        
    for view in VIEWS:
        # FIX: BASE_DIR already points to the root of the extracted files
        view_path = BASE_DIR / view 
        if not view_path.exists():
            print(f"⚠️ Warning: View path {view_path} not found. Skipping.")
            continue
        
        for family_folder in view_path.iterdir():
            if family_folder.is_dir():
                # Ignore metadata/COCO folders to focus on taxonomic complexity
                if "COCO" in family_folder.name: continue 
                
                target_family_dir = MERGED_DIR / family_folder.name
                target_family_dir.mkdir(exist_ok=True)
                
                for img in family_folder.glob('*'):
                    if img.is_file() and img.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif']:
                        # Prefix prevents collisions between different views
                        new_name = f"{view}_{img.name}"
                        shutil.copy2(img, target_family_dir / new_name)

    print(f"✅ Merge complete: {MERGED_DIR}")

    # 2. Split into Train (70%), Val (15%), Test (15%)
    splitfolders.ratio(
        str(MERGED_DIR), 
        output=str(SPLIT_DIR), 
        seed=42, 
        ratio=(.7, .15, .15), 
        move=False
    )
    print(f"✅ Split complete: {SPLIT_DIR}")

def count_dataset_distribution(root_dir):
    data = []
    phases = ['train', 'val', 'test']
    
    # Identify families from the processed train folder
    families = sorted([d.name for d in (root_dir / 'train').iterdir() if d.is_dir()])
    
    for family in families:
        row = {'Family': family}
        total_family = 0
        for phase in phases:
            phase_family_path = root_dir / phase / family
            count = len(list(phase_family_path.glob('*'))) if phase_family_path.exists() else 0
            row[phase] = count
            total_family += count
        
        row['Total'] = total_family
        data.append(row)
    
    df = pd.DataFrame(data)
    # Summarize totals for the final row
    totals = df.select_dtypes(include=['number']).sum()
    df_total = pd.DataFrame([{'Family': 'TOTAL SYSTEM', **totals}])
    return pd.concat([df, df_total], ignore_index=True)

if __name__ == "__main__":
    merge_and_split()
    
    # 3. Generate Report
    print("\n Dataset Distribution by family\:")
    distribution_df = count_dataset_distribution(SPLIT_DIR)
    print(distribution_df.to_string(index=False))