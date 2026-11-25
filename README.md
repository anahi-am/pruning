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

prune_yolo_model(
    model_path="yolo26n.pt",
    ratio=0.3,
    output_path="yolo26n_pruned.pt"
)
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