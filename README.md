# pruning

Structured pruning for Ultralytics YOLO models.

Removes filters from trained YOLO checkpoints to produce physically smaller,
faster models with minimal accuracy loss.

## Installation

```bash
uv add ultralytics torch torchvision
```

## Quickstart

```python
from src.pruner import prune
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
pruned = prune(model, ratio=0.3, method="magnitude", finetune=True, data="coco128.yaml")
```

## Project Structure

```
pruning/
├── src/
│   ├── __init__.py
│   ├── pruner.py          # core prune() logic
│   ├── scoring.py         # magnitude and gradient filter scoring
│   ├── graph.py           # dependency graph resolver
│   └── finetuner.py       # post-pruning fine-tune loop
├── benchmarks/
│   ├── run_benchmarks.py  # sparsity vs mAP vs speed
│   └── results/           # saved benchmark outputs
├── notebooks/
│   └── architecture_exploration.py
├── tests/
│   ├── test_graph.py
│   ├── test_scoring.py
│   └── test_pruner.py
├── pyproject.toml
└── README.md
```

## Benchmarks

| Sparsity | mAP before | mAP after | Speedup |
|---|---|---|---|
| 10% | - | - | - |
| 20% | - | - | - |
| 30% | - | - | - |
| 40% | - | - | - |
| 50% | - | - | - |

## References

- Han et al. (2015). *Learning both Weights and Connections for Efficient Neural Networks.*
- Molchanov et al. (2017). *Pruning Convolutional Neural Networks for Resource Efficient Inference.* 
