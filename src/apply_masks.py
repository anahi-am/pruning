import torch
import torch.nn as nn
from src.dependency_graph import DependencyGraph


def get_bn_from_conv(model: nn.Module, conv_name: str) -> nn.BatchNorm2d | None:
    parts = conv_name.split(".")
    bn_candidates = [".".join(parts[:-1] + ["bn"]), ".".join(parts + ["bn"])]

    for bn_name in bn_candidates:
        try:
            module = model.get_submodule(bn_name)
            if isinstance(module, nn.BatchNorm2d):
                return module
        except AttributeError:
            continue
    return None


def _get_layer_output_mask(
    layer_idx: int,
    model: nn.Module,
    masks: dict[str, torch.BoolTensor],
) -> torch.BoolTensor | None:
    """Return the output channel mask for a given layer index, or None if not pruned."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in masks:
            parts = name.split(".")
            try:
                if int(parts[1]) == layer_idx:
                    return masks[name]
            except IndexError, ValueError:
                continue
    return None


def _build_concat_input_mask(
    concat_layer_idx: int,
    source_indices: list[int],
    model: nn.Module,
    masks: dict[str, torch.BoolTensor],
    original_out_channels: dict[int, int],
) -> torch.BoolTensor:
    """Build a combined input mask for a concat layer from its source layers."""
    parts = []
    for src_idx in source_indices:
        src_mask = _get_layer_output_mask(src_idx, model, masks)
        orig = original_out_channels.get(src_idx)
        if src_mask is not None:
            parts.append(src_mask)
        elif orig is not None:
            parts.append(torch.ones(orig, dtype=torch.bool))
        else:
            return None
    if not parts:
        return None
    return torch.cat(parts)


def apply_masks(
    model: nn.Module,
    masks: dict[str, torch.BoolTensor],
    graph: DependencyGraph | None = None,
) -> nn.Module:
    if graph is None:
        graph = DependencyGraph()

    # Record original out_channels before any surgery
    original_out_channels: dict[int, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parts = name.split(".")
            try:
                layer_idx = int(parts[1])
                if layer_idx not in original_out_channels:
                    original_out_channels[layer_idx] = module.out_channels
            except IndexError, ValueError:
                continue

    # First pass: prune output channels of masked convs + their BNs
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or name not in masks:
            continue

        mask = masks[name]
        if not mask.any():
            continue

        module.weight = nn.Parameter(module.weight[mask])
        if module.bias is not None:
            module.bias = nn.Parameter(module.bias[mask])
        module.out_channels = int(mask.sum())

        bn = get_bn_from_conv(model, name)
        if bn:
            bn.num_features = int(mask.sum())
            if bn.running_mean is not None:
                bn.running_mean = bn.running_mean[mask]
            if bn.running_var is not None:
                bn.running_var = bn.running_var[mask]
            if bn.affine:
                if bn.weight is not None:
                    bn.weight = nn.Parameter(bn.weight[mask])
                if bn.bias is not None:
                    bn.bias = nn.Parameter(bn.bias[mask])

    # Second pass: prune input channels using dependency graph
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        parts = name.split(".")
        try:
            layer_idx = int(parts[1])
        except IndexError, ValueError:
            continue

        # Concat layer: input is concatenation of multiple source layers
        if layer_idx in graph.concat_map:
            source_indices = graph.concat_map[layer_idx]
            input_mask = _build_concat_input_mask(
                layer_idx, source_indices, model, masks, original_out_channels
            )
            if input_mask is not None and not input_mask.all():
                module.weight = nn.Parameter(module.weight[:, input_mask, :, :])
                module.in_channels = int(input_mask.sum())

        # Sequential layer: input comes from single predecessor
        else:
            src_idx = graph.get_input_source(layer_idx)
            if src_idx is None:
                continue
            src_mask = _get_layer_output_mask(src_idx, model, masks)
            if src_mask is not None and not src_mask.all():
                module.weight = nn.Parameter(module.weight[:, src_mask, :, :])
                module.in_channels = int(src_mask.sum())

    return model
