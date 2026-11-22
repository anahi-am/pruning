import torch
from ultralytics import YOLO


def magnitude_scores(model: YOLO) -> dict[str, torch.Tensor]:
    scores = {}
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data
            score = weight.abs().sum(dim=(1, 2, 3))
            scores[name] = score
    return scores