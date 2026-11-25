import torch
from torch import nn

from src.dependency_graph import DependencyGraph

DETECT_CV2_TEMPLATE = "model.{detect}.cv2.{scale}.0.conv"
DETECT_ONE2ONE_CV2_TEMPLATE = "model.{detect}.one2one_cv2.{scale}.0.conv"
DETECT_CV3_DEPTHWISE_TEMPLATE = "model.{detect}.cv3.{scale}.0.0.conv"
DETECT_CV3_NEXT_TEMPLATE = "model.{detect}.cv3.{scale}.0.1.conv"
DETECT_ONE2ONE_CV3_DEPTHWISE_TEMPLATE = "model.{detect}.one2one_cv3.{scale}.0.0.conv"
DETECT_ONE2ONE_CV3_NEXT_TEMPLATE = "model.{detect}.one2one_cv3.{scale}.0.1.conv"

DETECT_CV3_TEMPLATES = (
    (DETECT_CV3_DEPTHWISE_TEMPLATE, DETECT_CV3_NEXT_TEMPLATE),
    (DETECT_ONE2ONE_CV3_DEPTHWISE_TEMPLATE, DETECT_ONE2ONE_CV3_NEXT_TEMPLATE),
)


def get_bn_from_conv(model: nn.Module, conv_name: str) -> nn.BatchNorm2d | None:
    parts = conv_name.split(".")
    for bn_name in (".".join(parts[:-1] + ["bn"]), ".".join(parts + ["bn"])):
        try:
            module = model.get_submodule(bn_name)
        except AttributeError:
            continue
        if isinstance(module, nn.BatchNorm2d):
            return module
    return None


def _prune_batch_norm(bn: nn.BatchNorm2d, mask: torch.BoolTensor) -> None:
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


def _prune_conv_output(
    model: nn.Module, name: str, module: nn.Conv2d, mask: torch.BoolTensor
) -> None:
    if module.weight.shape[0] != mask.numel():
        raise ValueError(
            f"output mask length {mask.numel()} does not match out_channels "
            f"{module.weight.shape[0]} for {name}"
        )

    module.weight = nn.Parameter(module.weight[mask])
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias[mask])
    module.out_channels = int(mask.sum())

    bn = get_bn_from_conv(model, name)
    if bn is not None:
        _prune_batch_norm(bn, mask)


def _prune_conv_input(module: nn.Conv2d, mask: torch.BoolTensor) -> None:
    if module.weight.shape[1] != mask.numel():
        raise ValueError(
            f"input mask length {mask.numel()} does not match in_channels "
            f"{module.weight.shape[1]}"
        )
    module.weight = nn.Parameter(module.weight[:, mask, :, :])
    module.in_channels = int(mask.sum())


def _prune_depthwise(module: nn.Conv2d, mask: torch.BoolTensor) -> None:
    if module.groups != module.in_channels or module.groups != module.out_channels:
        raise ValueError("expected a depthwise convolution")
    if module.weight.shape[0] != mask.numel():
        raise ValueError(
            f"depthwise mask length {mask.numel()} does not match out_channels "
            f"{module.weight.shape[0]}"
        )

    module.weight = nn.Parameter(module.weight[mask])
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias[mask])
    kept = int(mask.sum())
    module.in_channels = kept
    module.out_channels = kept
    module.groups = kept


def _resolve_output_mask(
    idx: int,
    graph: DependencyGraph,
    masks: dict[str, torch.BoolTensor],
    original_out_channels: dict[int, int],
) -> torch.BoolTensor | None:
    conv_name = graph.get_output_conv_name(idx)
    if conv_name is not None:
        mask = masks.get(conv_name)
        if mask is not None:
            return mask
        original = original_out_channels.get(idx)
        if original is None:
            return None
        return torch.ones(original, dtype=torch.bool)

    sources = graph.get_sources(idx)
    if not sources:
        return None

    parts = []
    for source in sources:
        part = _resolve_output_mask(source, graph, masks, original_out_channels)
        if part is None:
            return None
        parts.append(part)
    return torch.cat(parts)


def _resolve_input_mask(
    idx: int,
    graph: DependencyGraph,
    masks: dict[str, torch.BoolTensor],
    original_out_channels: dict[int, int],
) -> torch.BoolTensor | None:
    sources = graph.get_sources(idx)
    if not sources:
        return None

    if len(sources) == 1:
        return _resolve_output_mask(sources[0], graph, masks, original_out_channels)

    parts = []
    for source in sources:
        part = _resolve_output_mask(source, graph, masks, original_out_channels)
        if part is None:
            return None
        parts.append(part)
    return torch.cat(parts)


def _apply_detect_masks(
    model: nn.Module,
    graph: DependencyGraph,
    masks: dict[str, torch.BoolTensor],
    original_out_channels: dict[int, int],
) -> None:
    if graph.detect_idx is None:
        return

    for scale, source in enumerate(graph.detect_sources):
        mask = _resolve_output_mask(source, graph, masks, original_out_channels)
        if mask is None or bool(mask.all()):
            continue

        cv2_name = DETECT_CV2_TEMPLATE.format(detect=graph.detect_idx, scale=scale)
        _prune_conv_input(model.get_submodule(cv2_name), mask)

        one2one_cv2_name = DETECT_ONE2ONE_CV2_TEMPLATE.format(
            detect=graph.detect_idx, scale=scale
        )
        _prune_conv_input(model.get_submodule(one2one_cv2_name), mask)

        for depthwise_template, next_template in DETECT_CV3_TEMPLATES:
            depthwise_name = depthwise_template.format(
                detect=graph.detect_idx, scale=scale
            )
            _prune_depthwise(model.get_submodule(depthwise_name), mask)

            bn = get_bn_from_conv(model, depthwise_name)
            if bn is not None:
                _prune_batch_norm(bn, mask)

            next_name = next_template.format(detect=graph.detect_idx, scale=scale)
            _prune_conv_input(model.get_submodule(next_name), mask)


def apply_masks(
    model: nn.Module,
    masks: dict[str, torch.BoolTensor],
    graph: DependencyGraph | None = None,
) -> nn.Module:
    if graph is None:
        graph = DependencyGraph(model)

    original_out_channels = {
        idx: channels
        for idx in range(len(graph.layers))
        if (channels := graph.get_original_out_channels(idx)) is not None
    }

    for name, mask in masks.items():
        module = model.get_submodule(name)
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"{name} is not a Conv2d")
        _prune_conv_output(model, name, module, mask)

    for idx in range(len(graph.layers)):
        if idx == graph.detect_idx:
            _apply_detect_masks(model, graph, masks, original_out_channels)
            continue

        entry_name = graph.get_entry_conv_name(idx)
        if entry_name is None:
            continue

        input_mask = _resolve_input_mask(idx, graph, masks, original_out_channels)
        if input_mask is None or bool(input_mask.all()):
            continue

        _prune_conv_input(model.get_submodule(entry_name), input_mask)

    return model
