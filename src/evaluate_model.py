# %% [markdown]
## Setup and Dependencies

# %%
!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

import os
from datetime import datetime
from ultralytics import YOLO

# UPDATE this to match your folder name in Google Drive
PROJECT_FOLDER_NAME = "pruning" 
PROJECT_DIR = f"/content/drive/MyDrive/{PROJECT_FOLDER_NAME}"
os.chdir(PROJECT_DIR)

MODELS = {
    "original": "yolo26n.pt",
    "pruned": "yolo26n_pruned.pt"
}

METRICS_FILE = "docs/metrics.md"

# %%
# Cell 2: Evaluation Function
def evaluate_model(name, path):
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found at {path}")
        return
        
    print(f"\n--- Evaluating {name} on COCO val2017 ---")
    model = YOLO(path)
    
    results = model.val(
        data="coco.yaml",
        imgsz=640,
        batch=32,
        device=0,
        verbose=True
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = (
        f"## Experiment: {name}\n"
        f"- **Date:** {timestamp}\n"
        f"- **Device:** GPU (Colab T4)\n"
        f"- **Dataset:** COCO val2017\n"
        f"- **Box mAP50:** {results.box.map50:.4f}\n"
        f"- **Box mAP50-95:** {results.box.map:.4f}\n"
        f"- **Precision:** {results.box.mp:.4f}\n"
        f"- **Recall:** {results.box.mr:.4f}\n\n"
    )
    
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    
    with open(METRICS_FILE, "a") as f:
        f.write(log_entry)
        
    print(f"Metrics saved to {METRICS_FILE}")
    print(log_entry)

# %%
# Cell 3: Run Evaluations
for name, path in MODELS.items():
    evaluate_model(name, path)

# %% [markdown]
# Evaluation complete. Check `docs/metrics.md` in your Google Drive for the final report.