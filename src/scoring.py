import torch
from ultralytics import YOLO

DEFAULT_SCORING = "l1"


def _l1(module: torch.nn.Conv2d) -> torch.Tensor:
    return module.weight.data.abs().sum(dim=(1, 2, 3))


def _batch_norm(model: YOLO, name: str) -> torch.nn.BatchNorm2d | None:
    parent = model.model.get_submodule(name.rsplit(".", 1)[0])
    bn = getattr(parent, "bn", None)
    return bn if isinstance(bn, torch.nn.BatchNorm2d) else None


def magnitude_scores(model: YOLO) -> dict[str, torch.Tensor]:
    return {
        name: _l1(module)
        for name, module in model.model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }


def bn_gamma_scores(model: YOLO) -> dict[str, torch.Tensor]:
    scores = {}
    for name, module in model.model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        bn = _batch_norm(model, name)
        scores[name] = bn.weight.data.abs() if bn is not None else _l1(module)
    return scores


def bn_l1_scores(model: YOLO) -> dict[str, torch.Tensor]:
    scores = {}
    for name, module in model.model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        l1 = _l1(module)
        bn = _batch_norm(model, name)
        scores[name] = bn.weight.data.abs() * l1 if bn is not None else l1
    return scores


SCORERS = {
    "l1": magnitude_scores,
    "bn": bn_gamma_scores,
    "bn_l1": bn_l1_scores,
}
SCORING_METHODS = tuple(SCORERS)


def compute_scores(
    model: YOLO, method: str = DEFAULT_SCORING
) -> dict[str, torch.Tensor]:
    if method not in SCORERS:
        raise ValueError(
            f"Unknown scoring method {method!r}; choose from {SCORING_METHODS}"
        )
    return SCORERS[method](model)
