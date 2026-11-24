import torch
import torch.nn as nn
from ultralytics import YOLO
from src.dependency_graph import DependencyGraph
from src.mask_generator import generate_pruning_masks
from src.apply_masks import apply_masks


def get_conv_names_for_layer(model: YOLO, layer_idx: int) -> list[str]:
    names = []
    sequential_block = model.model

    if not isinstance(sequential_block, nn.Sequential):
        return names

    target_module = sequential_block[layer_idx]

    for name, module in target_module.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            names.append(f"model.{layer_idx}.{name}")
    return names


def prune_yolo_model(
    model_path: str = "yolo26n.pt",
    ratio: float = 0.3,
    output_path: str = "yolo26n_pruned.pt",
) -> YOLO:
    model = YOLO(model_path)

    if not isinstance(model.model, nn.Module):
        raise ValueError("Model backbone is not loaded or is invalid")

    graph = DependencyGraph()

    raw_masks = generate_pruning_masks(model, ratio, graph)

    final_masks = {}
    for layer_idx_str, mask in raw_masks.items():
        try:
            layer_idx = int(layer_idx_str)
            conv_names = get_conv_names_for_layer(model, layer_idx)
            for conv_name in conv_names:
                final_masks[conv_name] = mask
        except ValueError:
            continue

    apply_masks(model.model, final_masks)

    model.save(output_path)
    print(f"Pruned model saved to {output_path}")

    return YOLO(output_path)


if __name__ == "__main__":
    prune_yolo_model()
