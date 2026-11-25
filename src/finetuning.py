# %% [markdown]
# # Fine-Tuning the Pruned YOLO Model on Full COCO
# Run this notebook in Kaggle with a GPU and Internet enabled.

# %%
!pip install -q ultralytics huggingface_hub

# %%
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.downloads import download

COCO_INPUT_DIR = Path("/kaggle/input/ms-coco-2017-dataset-yolo-format")
WORK_DATASET_DIR = Path("/kaggle/working/datasets/coco")
COCO_YAML = Path("/kaggle/working/coco_finetune.yaml")
OUTPUT_PATH = Path("/kaggle/working/yolo26n_pruned_finetuned.pt")
RUN_PROJECT = Path("/kaggle/working/runs/train")
RUN_NAME = "pruned_finetune"

MODEL_REPO_ID = "Anahia/yolo26n_pruned"
MODEL_FILENAME = "yolo26n_pruned.pt"

EPOCHS = 10
IMAGE_SIZE = 640
BATCH = 16
DEVICE = 0
WORKERS = 2
LR0 = 0.001
LRF = 0.01
WARMUP_EPOCHS = 3
PATIENCE = 5
AMP = True

EXPECTED_CLASSES = 80
LABEL_SAMPLE_SIZE = 20
SPLITS = ("train", "val")
ULTRALYTICS_SPLITS = ("train2017", "val2017")

# %%
def find_split_dir(root: Path, split: str) -> Path | None:
    candidates = [
        root / split,
        root / "images" / split,
        root / "coco2017" / split,
        root / "coco2017" / "images" / split,
        root / "coco" / split,
        root / "coco" / "images" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    for candidate in root.rglob(split):
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    return None


def find_label_dir(root: Path, split: str) -> Path | None:
    candidates = [
        root / "labels" / split,
        root / "coco" / "labels" / split,
        root / "coco2017" / "labels" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    for candidate in root.rglob(f"labels/{split}"):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def validate_labels(label_dir: Path, split: str) -> tuple[int, int]:
    files = list(label_dir.glob("*.txt"))[:LABEL_SAMPLE_SIZE]
    if not files:
        raise FileNotFoundError(f"No label files found in {label_dir}")

    min_class = EXPECTED_CLASSES
    max_class = -1
    for path in files:
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"{path} has {len(parts)} fields, expected 5 for detection boxes"
                )
            class_id = int(parts[0])
            min_class = min(min_class, class_id)
            max_class = max(max_class, class_id)

    if max_class < 0:
        raise ValueError(f"No annotations found in sampled {split} label files")

    if min_class < 0 or max_class >= EXPECTED_CLASSES:
        raise ValueError(
            f"{split} class ids range {min_class} to {max_class}, "
            f"expected 0 to {EXPECTED_CLASSES - 1}"
        )
    return min_class, max_class


# %%
if not COCO_INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Dataset root not found: {COCO_INPUT_DIR}")

direct_layout = all(
    (COCO_INPUT_DIR / split / "images").is_dir()
    and (COCO_INPUT_DIR / split / "labels").is_dir()
    for split in SPLITS
)

if direct_layout:
    DATA_ROOT = COCO_INPUT_DIR
    DATA_ENTRIES = {"train": "train/images", "val": "val/images"}
    LABEL_DIRS = {split: COCO_INPUT_DIR / split / "labels" for split in SPLITS}
    print(f"Using YOLO ready layout at {DATA_ROOT}")
else:
    image_dirs = {
        split: find_split_dir(COCO_INPUT_DIR, split) for split in ULTRALYTICS_SPLITS
    }
    missing_images = [split for split, path in image_dirs.items() if path is None]
    if missing_images:
        raise FileNotFoundError(
            f"Could not find images for {missing_images} under {COCO_INPUT_DIR}. "
            "Update COCO_INPUT_DIR to the mounted COCO dataset root"
        )

    label_dirs = {
        split: find_label_dir(COCO_INPUT_DIR, split) for split in ULTRALYTICS_SPLITS
    }
    yolo_ready = all(path is not None for path in label_dirs.values())

    WORK_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not yolo_ready:
        download(f"{ASSETS_URL}/coco2017labels.zip", dir=WORK_DATASET_DIR.parent, delete=True)
        label_dirs = {
            split: WORK_DATASET_DIR / "labels" / split for split in ULTRALYTICS_SPLITS
        }
        missing_labels = [
            split for split, path in label_dirs.items() if not path.is_dir()
        ]
        if missing_labels:
            raise FileNotFoundError(f"Downloaded labels are missing for {missing_labels}")

    for split in ULTRALYTICS_SPLITS:
        images_link = WORK_DATASET_DIR / "images" / split
        labels_link = WORK_DATASET_DIR / "labels" / split
        images_link.parent.mkdir(parents=True, exist_ok=True)
        labels_link.parent.mkdir(parents=True, exist_ok=True)
        if not images_link.exists():
            images_link.symlink_to(image_dirs[split], target_is_directory=True)
        if not labels_link.exists():
            labels_link.symlink_to(label_dirs[split], target_is_directory=True)

    DATA_ROOT = WORK_DATASET_DIR
    DATA_ENTRIES = {
        "train": f"images/{ULTRALYTICS_SPLITS[0]}",
        "val": f"images/{ULTRALYTICS_SPLITS[1]}",
    }
    LABEL_DIRS = {
        split: WORK_DATASET_DIR / "labels" / ultra
        for split, ultra in zip(SPLITS, ULTRALYTICS_SPLITS)
    }
    print(f"Normalized dataset root: {DATA_ROOT}")

# %%
config = YAML.load(check_yaml("coco.yaml"))
config["path"] = str(DATA_ROOT)
config["train"] = DATA_ENTRIES["train"]
config["val"] = DATA_ENTRIES["val"]
config.pop("test", None)
config.pop("download", None)
YAML.save(COCO_YAML, config)
print(f"Wrote dataset config to {COCO_YAML}")

# %%
for split in SPLITS:
    lowest, highest = validate_labels(LABEL_DIRS[split], split)
    print(f"{split} class ids: {lowest} to {highest}")

# %%
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = YOLO(model_path)
print(f"Loaded pruned model from {model_path}")
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

model.model.eval()
with torch.no_grad():
    model.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
print("Dummy forward pass succeeded")

# %%
results = model.train(
    data=str(COCO_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH,
    device=DEVICE,
    workers=WORKERS,
    lr0=LR0,
    lrf=LRF,
    warmup_epochs=WARMUP_EPOCHS,
    patience=PATIENCE,
    amp=AMP,
    project=str(RUN_PROJECT),
    name=RUN_NAME,
    exist_ok=True,
)

# %%
best_path = RUN_PROJECT / RUN_NAME / "weights" / "best.pt"
best_model = YOLO(str(best_path))
best_model.save(str(OUTPUT_PATH))
print(f"Saved fine-tuned model to {OUTPUT_PATH}")

# %%
metrics = YOLO(str(OUTPUT_PATH)).val(
    data=str(COCO_YAML),
    imgsz=IMAGE_SIZE,
    batch=BATCH,
    device=DEVICE,
)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50_95: {metrics.box.map:.4f}")
