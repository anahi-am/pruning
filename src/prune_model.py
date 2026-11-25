import torch
import torch.nn as nn
from ultralytics import YOLO
from src.dependency_graph import DependencyGraph
from src.mask_generator import generate_pruning_masks
from src.apply_masks import apply_masks


def prune_yolo_model(
    model_path: str = "yolo26n.pt",
    ratio: float = 0.3,
    output_path: str = "yolo26n_pruned.pt",
) -> YOLO:
    model = YOLO(model_path)

    if not isinstance(model.model, nn.Module):
        raise ValueError("Model backbone is not loaded or is invalid")

    graph = DependencyGraph()
    masks = generate_pruning_masks(model, ratio, graph)
    apply_masks(model.model, masks, graph)

    model.save(output_path)
    print(f"Pruned model saved to {output_path}")

    return YOLO(output_path)


if __name__ == "__main__":
    prune_yolo_model()
