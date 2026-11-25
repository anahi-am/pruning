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

| Sparsity | Model | mAP before | mAP after | Speedup |
| 0% | YOLO26n | - | - |
| 30% | YOLO26n-pruned | - | - | - |

See docs/metrics.md for detailed experimental logs

## Reproducing Results

To reproduce the full COCO validation metrics, use the provided evaluation script in Google Colab:
- Upload to Drive: Upload your entire project folder to the root of your Google Drive
- Open in Colab: Create a new notebook or open src/evaluate_model.py directly in Colab 
- Update Path: In the first code cell, update PROJECT_FOLDER_NAME to match your folder's name in Drive
- Run All Cells: The script will mount your Drive, navigate to the project, and run the full COCO validation on both the original and pruned models
