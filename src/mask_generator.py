import torch
from ultralytics import YOLO
from src.scoring import magnitude_scores
from src.dependency_graph import DependencyGraph


def generate_pruning_masks(
    model: YOLO, ratio: float, graph: DependencyGraph
) -> dict[str, torch.BoolTensor]:
    scores = magnitude_scores(model)
    masks = {}

    processed_layers = set()

    for name, score in scores.items():
        if name in processed_layers:
            continue

        layer_idx = _get_layer_index(name)
        if layer_idx is None:
            continue

        partners = graph.get_sync_partners(layer_idx)
        partner_names = _get_partner_names(model, partners)

        all_names = [name] + partner_names
        all_scores = [scores[n] for n in all_names if n in scores]

        if not all_scores:
            continue

        min_len = min(s.shape[0] for s in all_scores)
        trimmed_scores = [s[:min_len] for s in all_scores]

        avg_score = torch.mean(torch.stack(trimmed_scores), dim=0)

        num_to_keep = int(min_len * (1 - ratio))
        if num_to_keep == 0:
            num_to_keep = 1

        _, top_indices = torch.topk(avg_score, num_to_keep, largest=True)

        shared_mask = torch.zeros(min_len, dtype=torch.bool)
        shared_mask[top_indices] = True

        for n in all_names:
            if n in scores:
                full_mask = torch.zeros(scores[n].shape[0], dtype=torch.bool)
                full_mask[:min_len] = shared_mask
                full_mask[min_len:] = True
                masks[n] = full_mask
                processed_layers.add(n)

    return masks


def _get_layer_index(name: str) -> int | None:
    parts = name.split(".")
    try:
        return int(parts[1])
    except IndexError, ValueError:
        return None


def _get_partner_names(model: YOLO, partner_indices: set[int]) -> list[str]:
    names = []
    for idx in partner_indices:
        target_module = model.model[idx]
        for name, module in model.model.named_modules():
            if module is target_module:
                for sub_name, sub_module in target_module.named_modules():
                    if isinstance(sub_module, torch.nn.Conv2d) and "conv" in sub_name:
                        names.append(f"model.{idx}.{sub_name}")
                        break
                break
    return names
