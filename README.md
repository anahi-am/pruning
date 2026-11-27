# Structured pruning pipeline for Ultralytics YOLO models

Removes filters from trained YOLO checkpoints to produce physically smaller,
faster models with minimal accuracy loss.

## Installation

```bash
uv add ultralytics torch torchvision
```

## Quickstart

```python
from src.prune_model import prune_yolo_model

prune_yolo_model(model_path="yolo26n.pt", ratio=0.3, output_path="yolo26n_pruned.pt")
```

## Project Structure

```
pruning/
├── src/
│   ├── __init__.py
│   ├── prune_model.py     # Core coordination logic
│   ├── scoring.py         # L1-norm and activation-based scoring
│   ├── dependency_graph.py# Architectural dependency mapping
│   ├── mask_generator.py  # Synchronized mask creation
│   ├── apply_masks.py     # Weight surgery and BatchNorm sync
│   ├── fine_tuning.py     # Kaggle fine-tuning script
│   └── evaluate_model.py  # Colab-ready evaluation script
├── docs/
│   └── metrics.md         # Experimental results log
├── pyproject.toml
└── README.md
```

## Benchmarks

See docs/metrics.md for detailed experimental logs

## Reproducing Results

Run the evaluation in Google Colab against COCO val2017:

1. Open a new Colab notebook with a T4 GPU runtime
2. Paste the contents of `src/evaluate_model.py` and run all cells

The script pulls the original model from Ultralytics and the pruned checkpoint
from [Anahia/yolo26n_pruned](https://huggingface.co/Anahia/yolo26n_pruned) automatically.
No manual uploads required.

## Fine-Tuning on Kaggle

The pruned checkpoint can be fine-tuned on full COCO with `src/fine_tuning.py`.

1. Create a Kaggle notebook with a GPU accelerator and Internet enabled
2. Add the dataset `alifshahariar/ms-coco-2017-dataset-yolo-format` via Add Input
3. Paste the contents of `src/fine_tuning.py` into the notebook and run the cells in order

The dataset is already in YOLO format and mounts at
`/kaggle/input/ms-coco-2017-dataset-yolo-format`:

```
train/
  images/
  labels/
val/
  images/
  labels/
```

The script:

- builds `/kaggle/working/coco_finetune.yaml` from the dataset layout
- validates sampled labels (5 fields, class IDs 0 to 79) before training
- pulls `Anahia/yolo26n_pruned` from Hugging Face
- trains for `EPOCHS` epochs and writes `/kaggle/working/yolo26n_pruned_finetuned.pt`
- runs `val()` on the fine-tuned checkpoint

Outputs:

- `/kaggle/working/runs/train/pruned_finetune/weights/best.pt`
- `/kaggle/working/yolo26n_pruned_finetuned.pt`

Notes:

- `/kaggle/input` is read only. Ultralytics skips its label cache and rescans labels on each run
- 10 epochs on full COCO can approach the 12 hour session limit. Lower `EPOCHS` or use a subset if needed
- if the label check fails, the dataset class IDs do not match the 80 class COCO order

## Tests

```bash
uv run pytest -q
```

Integration tests use the real `yolo26n.pt` weights and are marked `integration`.
