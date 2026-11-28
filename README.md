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

prune_yolo_model(model_path="yolo26n.pt", ratio=0.1, output_path="yolo26n_pruned.pt")
```

Defaults are `scoring="l1"` and `layer_ratios="depth"`.

## Scoring

The ratio decides how many channels to remove per layer; the scoring method
decides which channels survive. Three methods are available:

- `l1` (default): rank filters by L1 magnitude, `|W|.sum(dim=(1, 2, 3))`. Needs
  no calibration data and is deterministic.
- `bn`: rank filters by the scale (gamma) of the BatchNorm that follows each
  convolution.
- `bn_l1`: multiply `|gamma|` by the filter L1 norm (Network Slimming).

`bn` and `bn_l1` fall back to L1 for convolutions without BatchNorm.

```python
prune_yolo_model(..., scoring="bn_l1")
```

```bash
python -m src.prune_model --ratio 0.1 --scoring bn_l1
```

On coco8 with uniform layer ratios at ratio 0.05, L1 beat BN gamma (untuned
mAP50 0.2845 vs 0.0184) and BN-L1 (0.1658); at 0.10 all three collapsed. L1
remains the default. See docs/pruning-decisions.md for the full comparison.

## Layer Ratios

`layer_ratios` controls how the ratio is distributed across the network:

- `uniform`: apply the same ratio to every tracked output convolution.
- `depth` (default): exempt the first tracked convolution and the Detect head,
  then ramp the ratio linearly from 0 at the first tracked convolution to
  `ratio` at the deepest.

```python
prune_yolo_model(..., layer_ratios="uniform")
```

```bash
python -m src.prune_model --ratio 0.1 --layer-ratios uniform
```

On coco8 with L1 scoring:

| ratio | layer ratios | params    | untuned mAP50 | 1-epoch mAP50 |
|-------|--------------|-----------|---------------|---------------|
| 0.05  | uniform      | 2,444,808 | 0.2845        | 0.2954        |
| 0.05  | depth        | 2,506,030 | 0.8419        | 0.8471        |
| 0.10  | uniform      | 2,332,312 | 0.0039        | 0.0045        |
| 0.10  | depth        | 2,447,937 | 0.4529        | 0.5087        |

`depth` is the default because it recovers far more accuracy at both ratios,
especially 0.10. It prunes fewer parameters because it protects early layers:
at 0.10 it removes about 124k parameters (4.8%) versus about 240k (9.3%) for
`uniform`.

## Project Structure

```
pruning/
├── src/
│   ├── __init__.py
│   ├── prune_model.py     # Core coordination logic
│   ├── scoring.py         # L1, BatchNorm-gamma, and combined scoring
│   ├── dependency_graph.py# Architectural dependency mapping
│   ├── mask_generator.py  # Synchronized mask creation and layer ratios
│   ├── apply_masks.py     # Weight surgery and BatchNorm sync
│   ├── fine_tuning.py     # Kaggle fine-tuning script
│   ├── fine_tuning_smoke.py # Colab smoke test
│   └── evaluate_model.py  # Colab-ready evaluation script
├── docs/
│   ├── project-direction.md  # Use cases and model goals
│   ├── pruning-decisions.md  # Pruning ratio and scoring decisions
│   └── smoke-test.md         # Fine-tuning smoke test report and plan
├── pyproject.toml
└── README.md
```

## Benchmarks

See docs/pruning-decisions.md for the pruning ratio decision log

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
- trains the loaded pruned module directly, bypassing the `model.yaml` rebuild
  that would otherwise reconstruct the unpruned architecture
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
