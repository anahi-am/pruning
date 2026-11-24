import torch
import torch.nn as nn


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


def apply_masks(model: nn.Module, masks: dict[str, torch.BoolTensor]) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in masks:
            mask = masks[name]

            if not mask.any():
                continue

            new_weight = module.weight[mask]
            module.weight = nn.Parameter(new_weight)

            if module.bias is not None:
                new_bias = module.bias[mask]
                module.bias = nn.Parameter(new_bias)

            module.out_channels = int(mask.sum())

            bn_layer = get_bn_from_conv(model, name)
            if bn_layer:
                bn_layer.num_features = int(mask.sum())

                if bn_layer.running_mean is not None:
                    bn_layer.running_mean = bn_layer.running_mean[mask]
                if bn_layer.running_var is not None:
                    bn_layer.running_var = bn_layer.running_var[mask]

                if bn_layer.affine:
                    if bn_layer.weight is not None:
                        bn_layer.weight = nn.Parameter(bn_layer.weight[mask])
                    if bn_layer.bias is not None:
                        bn_layer.bias = nn.Parameter(bn_layer.bias[mask])

    return model
