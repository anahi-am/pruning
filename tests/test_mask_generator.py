import pytest
import torch

from src.dependency_graph import DependencyGraph
from src.mask_generator import generate_pruning_masks
from tests.helpers import FakeYOLO, TinyC3k2, TinyConv, TinyModel


def build_model() -> TinyModel:
    return TinyModel([TinyConv(4, 8), TinyC3k2(8, 8)])


def test_masks_only_output_convs() -> None:
    detection_model = build_model()
    graph = DependencyGraph(detection_model)

    masks = generate_pruning_masks(FakeYOLO(detection_model), 0.5, graph)

    assert set(masks) == {"model.0.conv", "model.1.cv2.conv"}
    assert "model.1.cv1.conv" not in masks
    for mask in masks.values():
        assert mask.dtype == torch.bool
        assert int(mask.sum()) == 4


def test_ratio_validation() -> None:
    detection_model = build_model()
    graph = DependencyGraph(detection_model)

    with pytest.raises(ValueError):
        generate_pruning_masks(FakeYOLO(detection_model), 0.0, graph)
    with pytest.raises(ValueError):
        generate_pruning_masks(FakeYOLO(detection_model), 1.0, graph)
