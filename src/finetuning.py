# %% [markdown]
# # Fine-Tuning the Pruned YOLO Model on Full COCO
# Run this notebook in Kaggle with a GPU and Internet enabled.

# %%
import subprocess
import sys
from pathlib import Path

if Path("/kaggle").is_dir():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "huggingface_hub"],
        check=True,
    )

# %%
from dataclasses import dataclass

import torch
from ultralytics import YOLO
try:
    from ultralytics.utils import ASSETS_URL
except ImportError:
    from ultralytics.utils.downloads import ASSETS_URL
from ultralytics.utils import YAML
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
PATIENCE = 10
AMP = True

EXPECTED_CLASSES = 80
LABEL_SAMPLE_SIZE = 200
SPLITS = ("train", "val")
ULTRALYTICS_SPLITS = ("train2017", "val2017")

# %%
@dataclass(frozen=True)
class DatasetLayout:
    data_root: Path
    train_entry: str
    val_entry: str
    label_dirs: dict[str, Path]
    uses_input: bool


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
        root / split / "labels",
        root / "coco" / "labels" / split,
        root / "coco2017" / "labels" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    for candidate in root.rglob(f"labels/{split}"):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    for candidate in root.rglob(f"{split}/labels"):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def resolve_dataset_layout(input_dir: Path, work_dir: Path) -> DatasetLayout:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {input_dir}")

    direct = all(
        (input_dir / split / "images").is_dir()
        and (input_dir / split / "labels").is_dir()
        for split in SPLITS
    )
    if direct:
        return DatasetLayout(
            data_root=input_dir,
            train_entry="train/images",
            val_entry="val/images",
            label_dirs={split: input_dir / split / "labels" for split in SPLITS},
            uses_input=True,
        )

    image_dirs = {
        split: find_split_dir(input_dir, split) for split in ULTRALYTICS_SPLITS
    }
    missing_images = [split for split, path in image_dirs.items() if path is None]
    if missing_images:
        raise FileNotFoundError(
            f"Could not find images for {missing_images} under {input_dir}. "
            "Update COCO_INPUT_DIR to the mounted COCO dataset root"
        )
    image_dirs = {split: path for split, path in image_dirs.items() if path is not None}

    label_dirs = {
        split: find_label_dir(input_dir, ultra)
        for split, ultra in zip(SPLITS, ULTRALYTICS_SPLITS)
    }
    if all(path is not None for path in label_dirs.values()):
        label_dirs = {
            split: path for split, path in label_dirs.items() if path is not None
        }
        return DatasetLayout(
            data_root=input_dir,
            train_entry=f"images/{ULTRALYTICS_SPLITS[0]}",
            val_entry=f"images/{ULTRALYTICS_SPLITS[1]}",
            label_dirs=label_dirs,
            uses_input=True,
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    download(f"{ASSETS_URL}/coco2017labels.zip", dir=work_dir.parent, delete=True)
    downloaded_labels = {
        split: work_dir / "labels" / ultra
        for split, ultra in zip(SPLITS, ULTRALYTICS_SPLITS)
    }
    missing_labels = [
        split for split, path in downloaded_labels.items() if not path.is_dir()
    ]
    if missing_labels:
        raise FileNotFoundError(f"Downloaded labels are missing for {missing_labels}")

    for split, ultra in zip(SPLITS, ULTRALYTICS_SPLITS):
        images_link = work_dir / "images" / ultra
        labels_link = work_dir / "labels" / ultra
        images_link.parent.mkdir(parents=True, exist_ok=True)
        labels_link.parent.mkdir(parents=True, exist_ok=True)
        if not images_link.is_symlink() and not images_link.exists():
            images_link.symlink_to(image_dirs[ultra], target_is_directory=True)
        if not labels_link.is_symlink() and not labels_link.exists():
            labels_link.symlink_to(downloaded_labels[split], target_is_directory=True)

    return DatasetLayout(
        data_root=work_dir,
        train_entry=f"images/{ULTRALYTICS_SPLITS[0]}",
        val_entry=f"images/{ULTRALYTICS_SPLITS[1]}",
        label_dirs=downloaded_labels,
        uses_input=False,
    )


def validate_labels(label_dir: Path, split: str) -> tuple[int, int]:
    files = list(label_dir.glob("*.txt"))[:LABEL_SAMPLE_SIZE]
    if not files:
        raise FileNotFoundError(f"No label files found in {label_dir}")

    min_class = EXPECTED_CLASSES
    max_class = -1
    non_empty = 0
    for path in files:
        has_annotation = False
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            has_annotation = True
            if len(parts) != 5:
                raise ValueError(
                    f"{path} has {len(parts)} fields, expected 5 for detection boxes"
                )
            class_id = int(parts[0])
            min_class = min(min_class, class_id)
            max_class = max(max_class, class_id)
        if has_annotation:
            non_empty += 1

    if non_empty == 0:
        raise ValueError(f"No annotations found in sampled {split} label files")

    if min_class < 0 or max_class >= EXPECTED_CLASSES:
        raise ValueError(
            f"{split} class ids range {min_class} to {max_class}, "
            f"expected 0 to {EXPECTED_CLASSES - 1}"
        )
    return min_class, max_class


def build_dataset_config(
    data_root: Path,
    train_entry: str,
    val_entry: str,
    output_path: Path,
) -> dict:
    config = YAML.load(check_yaml("coco.yaml"))
    config["path"] = str(data_root)
    config["train"] = train_entry
    config["val"] = val_entry
    config.pop("test", None)
    config.pop("download", None)
    YAML.save(output_path, config)
    return config


# %%
if __name__ == "__main__":
    layout = resolve_dataset_layout(COCO_INPUT_DIR, WORK_DATASET_DIR)
    print(f"Using dataset root {layout.data_root}")

    build_dataset_config(
        layout.data_root, layout.train_entry, layout.val_entry, COCO_YAML
    )
    print(f"Wrote dataset config to {COCO_YAML}")

    for split in SPLITS:
        lowest, highest = validate_labels(layout.label_dirs[split], split)
        print(f"{split} class ids: {lowest} to {highest}")

# %%
if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = YOLO(model_path)
    print(f"Loaded pruned model from {model_path}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    model.model.eval()
    with torch.no_grad():
        model.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    print("Dummy forward pass succeeded")

# %%
if __name__ == "__main__":
    model.train(
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
if __name__ == "__main__":
    best_path = RUN_PROJECT / RUN_NAME / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    best_model.save(str(OUTPUT_PATH))
    print(f"Saved fine-tuned model to {OUTPUT_PATH}")

# %%
if __name__ == "__main__":
    metrics = YOLO(str(OUTPUT_PATH)).val(
        data=str(COCO_YAML),
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50_95: {metrics.box.map:.4f}")
