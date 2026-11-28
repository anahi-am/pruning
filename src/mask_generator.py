import torch
from ultralytics import YOLO

from src.dependency_graph import DependencyGraph
from src.scoring import DEFAULT_SCORING, compute_scores

MIN_KEPT_CHANNELS = 1
LAYER_RATIO_MODES = ("uniform", "depth")
DEFAULT_LAYER_RATIOS = "depth"


def _resolve_layer_ratio(
    ratio: float,
    idx: int,
    first_idx: int,
    max_idx: int,
    detect_idx: int | None,
    mode: str,
) -> float:
    if mode == "uniform":
        return ratio
    if idx == first_idx or idx == detect_idx:
        return 0.0
    span = max_idx - first_idx
    if span <= 0:
        return ratio
    return ratio * (idx - first_idx) / span


def generate_pruning_masks(
    model: YOLO,
    ratio: float,
    graph: DependencyGraph,
    scoring: str = DEFAULT_SCORING,
    layer_ratios: str = DEFAULT_LAYER_RATIOS,
) -> dict[str, torch.BoolTensor]:
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    if layer_ratios not in LAYER_RATIO_MODES:
        raise ValueError(
            f"Unknown layer ratio mode {layer_ratios!r}; "
            f"choose from {LAYER_RATIO_MODES}"
        )

    scores = compute_scores(model, scoring)
    tracked = []
    for idx in range(len(graph.layers)):
        conv_name = graph.get_output_conv_name(idx)
        if conv_name in scores:
            tracked.append((idx, conv_name))

    if not tracked:
        return {}

    first_idx = tracked[0][0]
    max_idx = tracked[-1][0]
    masks: dict[str, torch.BoolTensor] = {}

    for idx, conv_name in tracked:
        score = scores[conv_name]
        layer_ratio = _resolve_layer_ratio(
            ratio, idx, first_idx, max_idx, graph.detect_idx, layer_ratios
        )
        num_to_keep = max(MIN_KEPT_CHANNELS, int(score.shape[0] * (1 - layer_ratio)))
        _, top_indices = torch.topk(score, num_to_keep, largest=True)

        mask = torch.zeros(score.shape[0], dtype=torch.bool)
        mask[top_indices] = True
        masks[conv_name] = mask

    return masks
