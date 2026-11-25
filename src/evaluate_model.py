# %%
!pip install ultralytics huggingface_hub

from huggingface_hub import hf_hub_download
import os
from datetime import datetime
from ultralytics import YOLO

MODELS = {
    "original": "yolo26n.pt",
    "pruned": hf_hub_download(repo_id="Anahia/yolo26n_pruned", filename="yolo26n_pruned.pt"),
}

METRICS_FILE = "docs/metrics.md"


# %%
def evaluate_model(name: str, path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"Skipping {name}: file not found at {path}")
        return None

    print(f"\n--- Evaluating {name} on COCO val2017 ---")
    model = YOLO(path)

    results = model.val(
        data="coco.yaml",
        imgsz=640,
        batch=32,
        device=0,
        verbose=True,
    )

    return {
        "name": name,
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "inference_ms": results.speed["inference"],
    }


# %%
os.makedirs("docs", exist_ok=True)

for name, path in MODELS.items():
    r = evaluate_model(name, path)
    if r is None:
        continue

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"## Experiment: {r['name']}\n"
        f"- **Date:** {timestamp}\n"
        f"- **Device:** GPU (Colab T4)\n"
        f"- **Dataset:** COCO val2017\n"
        f"- **Box mAP50:** {r['map50']:.4f}\n"
        f"- **Box mAP50-95:** {r['map50_95']:.4f}\n"
        f"- **Precision:** {r['precision']:.4f}\n"
        f"- **Recall:** {r['recall']:.4f}\n"
        f"- **Inference (ms):** {r['inference_ms']:.2f}\n\n"
    )

    with open(METRICS_FILE, "a") as f:
        f.write(entry)
    print(entry)

# %%
from google.colab import files
files.download("docs/metrics.md")