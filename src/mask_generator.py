import torch
from ultralytics import YOLO

from src.dependency_graph import DependencyGraph
from src.scoring import magnitude_scores

MIN_KEPT_CHANNELS = 1


def generate_pruning_masks(
    model: YOLO, ratio: float, graph: DependencyGraph
) -> dict[str, torch.BoolTensor]:
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    scores = magnitude_scores(model)
    masks: dict[str, torch.BoolTensor] = {}

    for idx in range(len(graph.layers)):
        conv_name = graph.get_output_conv_name(idx)
        if conv_name is None or conv_name not in scores:
            continue

        score = scores[conv_name]
        num_to_keep = max(MIN_KEPT_CHANNELS, int(score.shape[0] * (1 - ratio)))
        _, top_indices = torch.topk(score, num_to_keep, largest=True)

        mask = torch.zeros(score.shape[0], dtype=torch.bool)
        mask[top_indices] = True
        masks[conv_name] = mask

    return masks
