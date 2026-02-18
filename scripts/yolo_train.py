import os
import gc
import torch
import shutil
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
# Path to your already divided dataset (classification format)
DATA_ROOT = Path('data/processed/DAPWH_Final_Split')
# Testing the SOTA extra-large variants
MODELS_TO_TRAIN = ['yolov12n-cls', 'yolo26n-cls'] 
RESULTS_DIR = Path('reports/model_comparison')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def clear_gpu_memory():
    """Clears VRAM to prevent fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("🧹 GPU memory cache cleared.")

def train_and_evaluate():
    summary_data = []

    for model_name in MODELS_TO_TRAIN:
        print(f"\n🚀 Starting Training: {model_name}")
        
        # 1. Load Model
        model = YOLO(f"{model_name}.pt")
        
        # 2. Train Model
        train_results = model.train(
            data=DATA_ROOT,
            epochs=150,
            imgsz=512,
            batch=32,
            name=f"DAPWH_{model_name}",
            exist_ok=True
        )

        # 3. Final Evaluation on Test Set
        print(f"📊 Evaluating {model_name} on Final Test Set...")
        metrics = model.val(split='test', plots=True) 

        # --- FIX: Manual Metric Extraction for Classification ---
        # Accessing the internal confusion matrix for Precision/Recall calculation
        cm = metrics.confusion_matrix.matrix
        
        # Rows = True, Cols = Predicted
        tp = cm.diagonal()
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        eps = 1e-7 # Prevent division by zero
        precision_per_class = tp / (tp + fp + eps)
        recall_per_class = tp / (tp + fn + eps)
        
        mean_precision = precision_per_class.mean()
        mean_recall = recall_per_class.mean()
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + eps)

        # 4. Export Confusion Matrix Image
        cm_src = Path(train_results.save_dir) / 'confusion_matrix.png'
        cm_dest = RESULTS_DIR / f"CM_{model_name}_test.png"
        if cm_src.exists():
            shutil.copy(cm_src, cm_dest)
            print(f" Confusion Matrix saved to: {cm_dest}")

        # 5. Append Results
        summary_data.append({
            'Model': model_name,
            'Accuracy (Top1)': round(metrics.top1, 4),
            'Precision': round(mean_precision, 4),
            'Recall': round(mean_recall, 4),
            'F1-Score': round(f1_score, 4),
            'Fitness': round(metrics.fitness, 4)
        })
        
        # Cleanup for next iteration
        del model
        clear_gpu_memory()

    # 6. Save Comparison Report
    df = pd.DataFrame(summary_data)
    df.to_csv(RESULTS_DIR / 'model_performance_summary.csv', index=False)
    
    print("\n Final Comparison Table (Test Set Results):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()