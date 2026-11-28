# %% [markdown]
# # Fine-Tuning Smoke Test — COCO val2017 (5k images)
# Run this notebook in Colab with a GPU runtime.
# Purpose: validate the training loop end-to-end before the full Kaggle run.
# A few epochs on 5k images gives meaningful recovery signal in ~25 minutes.

# %%
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "huggingface_hub"],
    check=True,
)

# %%
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from huggingface_hub import hf_hub_download

WORK_DIR = Path("/content/datasets/coco_smoke")
COCO_YAML = Path("/content/coco_smoke.yaml")
OUTPUT_PATH = Path("/content/yolo26n_pruned_smoke_finetuned.pt")
RUN_PROJECT = Path("/content/runs/train")
RUN_NAME = "smoke_finetune"

MODEL_REPO_ID = "Anahia/yolo26n_pruned"
MODEL_FILENAME = "yolo26n_pruned.pt"

EPOCHS = 5
IMAGE_SIZE = 640
BATCH = 16
DEVICE = 0
WORKERS = 2
LR0 = 0.001
LRF = 0.01
WARMUP_EPOCHS = 1
PATIENCE = 5
AMP = True

SANITY_DATA = "coco8.yaml"
SANITY_MIN_MAP50 = 0.001

# %% [markdown]
# ## 1. Download COCO val2017 as training data
# val2017 = 5k images, same 80 classes as full COCO, available via Ultralytics CDN.
# We use it as both train and val for the smoke test — the goal is pipeline
# validation, not generalisation metrics.

# %%
import zipfile
import urllib.request

WORK_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
)

images_zip = Path("/content/val2017.zip")
labels_zip = Path("/content/coco2017labels.zip")

if not images_zip.exists():
    print("Downloading val2017 images (~800MB)...")
    urllib.request.urlretrieve(IMAGES_URL, images_zip)

if not labels_zip.exists():
    print("Downloading labels...")
    urllib.request.urlretrieve(LABELS_URL, labels_zip)

print("Extracting images...")
with zipfile.ZipFile(images_zip) as z:
    z.extractall("/content/datasets/coco_smoke/images/")

print("Extracting labels...")
with zipfile.ZipFile(labels_zip) as z:
    z.extractall("/content/datasets/")

# Ultralytics labels zip unpacks to coco/labels/val2017
LABEL_DIR = Path("/content/datasets/coco/labels/val2017")
IMAGE_DIR = WORK_DIR / "images" / "val2017"

print(f"Images: {len(list(IMAGE_DIR.glob('*.jpg')))} found")
print(f"Labels: {len(list(LABEL_DIR.glob('*.txt')))} found")

# %%
config = YAML.load(check_yaml("coco.yaml"))
config["path"] = str(WORK_DIR)
config["train"] = "images/val2017"
config["val"] = "images/val2017"
config.pop("test", None)
config.pop("download", None)

# point labels path so Ultralytics finds them
# Ultralytics infers label path by replacing /images/ with /labels/ in the image path
# so labels must live at WORK_DIR/labels/val2017
labels_link = WORK_DIR / "labels" / "val2017"
labels_link.parent.mkdir(parents=True, exist_ok=True)
if not labels_link.exists():
    labels_link.symlink_to(LABEL_DIR, target_is_directory=True)

YAML.save(COCO_YAML, config)
print(f"Dataset config written to {COCO_YAML}")

# %% [markdown]
# ## 2. Load pruned model

# %%
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = YOLO(model_path)
print(f"Loaded pruned model from {model_path}")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

model.model.eval()
with torch.no_grad():
    model.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
print("Dummy forward pass succeeded")

# %% [markdown]
# ## 3. Sanity check before training

# %%
metrics = model.val(data=SANITY_DATA, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)
sanity_map50 = float(metrics.box.map50)
if sanity_map50 < SANITY_MIN_MAP50:
    raise RuntimeError(
        f"Sanity check failed: mAP50 {sanity_map50:.4f} on {SANITY_DATA} "
        f"is below {SANITY_MIN_MAP50}; checkpoint is likely broken"
    )
print(f"Sanity check passed: mAP50 {sanity_map50:.4f} on {SANITY_DATA}")

# %% [markdown]
# ## 4. Fine-tune

# %%
trainer = DetectionTrainer(
    overrides={
        "model": model_path,
        "data": str(COCO_YAML),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "lr0": LR0,
        "lrf": LRF,
        "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE,
        "amp": AMP,
        "project": str(RUN_PROJECT),
        "name": RUN_NAME,
        "exist_ok": True,
    }
)
expected_params = sum(p.numel() for p in model.model.parameters())
model.model.requires_grad_(True)
trainer.model = model.model
trainer.setup_model()
actual_params = sum(p.numel() for p in trainer.model.parameters())
if actual_params != expected_params:
    raise RuntimeError(
        f"Trainer rebuilt the model from model.yaml: {actual_params:,} parameters, "
        f"expected the pruned {expected_params:,}"
    )
trainer.train()

# %% [markdown]
# ## 5. Save and verify

# %%
best_path = RUN_PROJECT / RUN_NAME / "weights" / "best.pt"
best_model = YOLO(str(best_path))
best_model.save(str(OUTPUT_PATH))

reloaded = YOLO(str(OUTPUT_PATH))
reloaded.model.eval()
with torch.no_grad():
    reloaded.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
print(f"Saved fine-tuned model to {OUTPUT_PATH}")
print("Post-training reload verified")

# %% [markdown]
# ## 6. Evaluate

# %%
metrics = YOLO(str(OUTPUT_PATH)).val(
    data=str(COCO_YAML),
    imgsz=IMAGE_SIZE,
    batch=BATCH,
    device=DEVICE,
)
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print()
print("Smoke test complete. If mAP50 is trending upward across epochs,")
print("the training loop is working. Proceed to the full Kaggle run.")
