import argparse
import copy

import torch
from torch import nn
from ultralytics import YOLO

from src.apply_masks import apply_masks
from src.dependency_graph import DependencyGraph
from src.mask_generator import generate_pruning_masks

DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_RATIO = 0.3
DEFAULT_OUTPUT = "yolo26n_pruned.pt"
DEFAULT_IMAGE_SIZE = 640


def prune_yolo_model(
    model_path: str = DEFAULT_MODEL,
    ratio: float = DEFAULT_RATIO,
    output_path: str = DEFAULT_OUTPUT,
    dry_run: bool = False,
) -> YOLO:
    model = YOLO(model_path)
    detection_model = model.model

    if not isinstance(detection_model, nn.Module):
        raise TypeError("Model backbone is not loaded or is invalid")

    graph = DependencyGraph(detection_model)
    masks = generate_pruning_masks(model, ratio, graph)

    if dry_run:
        _report_dry_run(detection_model, graph, masks, ratio)
        return model

    apply_masks(detection_model, masks, graph)
    _validate_forward(detection_model)

    model.save(output_path)
    print(f"Pruned model saved to {output_path}")

    return YOLO(output_path)


def _validate_forward(detection_model: nn.Module) -> None:
    detection_model.eval()
    with torch.no_grad():
        detection_model(torch.zeros(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))


def _report_dry_run(
    detection_model: nn.Module,
    graph: DependencyGraph,
    masks: dict[str, torch.BoolTensor],
    ratio: float,
) -> None:
    print(f"Dry run: ratio={ratio}")
    print(
        f"{'layer':>5} {'type':<12} {'output conv':<28} "
        f"{'orig':>6} {'kept':>6} {'removed':>8}"
    )

    for idx in range(len(graph.layers)):
        conv_name = graph.get_output_conv_name(idx)
        if conv_name is None or conv_name not in masks:
            continue
        mask = masks[conv_name]
        original = int(mask.numel())
        kept = int(mask.sum())
        removed = 100.0 * (original - kept) / original
        print(
            f"{idx:>5} {type(graph.layers[idx]).__name__:<12} {conv_name:<28} "
            f"{original:>6} {kept:>6} {removed:>7.1f}%"
        )

    before = sum(p.numel() for p in detection_model.parameters())
    trial = copy.deepcopy(detection_model)
    try:
        apply_masks(trial, masks, DependencyGraph(trial))
        _validate_forward(trial)
    except Exception as exc:
        print(f"Dry run failed: {exc}")
        raise SystemExit(1) from exc

    after = sum(p.numel() for p in trial.parameters())
    reduction = 100.0 * (before - after) / before
    print(f"Params: {before:,} -> {after:,} ({reduction:.2f}% reduction)")
    print("Dry run passed: masks are structurally consistent")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured pruning for Ultralytics YOLO models"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prune_yolo_model(args.model, args.ratio, args.output, args.dry_run)


if __name__ == "__main__":
    main()
