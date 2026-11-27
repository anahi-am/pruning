import torch

from src.apply_masks import apply_masks
from src.dependency_graph import DependencyGraph
from tests.helpers import (
    Concat,
    Detect,
    TinyC3k2,
    TinyConv,
    TinyModel,
    TinySPPF,
    make_mask,
)


def build_model() -> TinyModel:
    return TinyModel(
        [
            TinyConv(4, 8),
            TinyC3k2(8, 8),
            TinyConv(8, 8),
            Concat([-1, 1]),
            TinyConv(16, 8),
            Concat([-1, 0]),
            TinyConv(16, 4),
        ]
    )


def test_apply_masks_preserves_internal_structure_and_forwards() -> None:
    model = build_model()
    graph = DependencyGraph(model)
    masks = {
        "model.0.conv": make_mask(8, 6),
        "model.1.cv2.conv": make_mask(8, 6),
        "model.2.conv": make_mask(8, 7),
        "model.4.conv": make_mask(8, 6),
        "model.6.conv": make_mask(4, 3),
    }

    apply_masks(model, masks, graph)

    assert model.model[1].cv1.conv.in_channels == 6
    assert model.model[1].cv1.conv.out_channels == 8
    assert model.model[1].cv2.conv.in_channels == 16
    assert model.model[1].cv2.conv.out_channels == 6
    assert model.model[2].conv.in_channels == 6
    assert model.model[2].conv.out_channels == 7
    assert model.model[4].conv.in_channels == 13
    assert model.model[4].conv.out_channels == 6
    assert model.model[6].conv.in_channels == 12
    assert model.model[6].conv.out_channels == 3

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 4, 8, 8))
    assert output.shape == (1, 3, 8, 8)


def test_residual_output_mask_follows_input_mask() -> None:
    model = TinyModel([TinyConv(4, 4), TinySPPF(4), TinyConv(4, 4)])
    graph = DependencyGraph(model)
    input_mask = torch.tensor([True, True, True, False])
    conflicting_output_mask = torch.tensor([False, True, True, True])
    masks = {
        "model.0.conv": input_mask,
        "model.1.cv2.conv": conflicting_output_mask,
        "model.2.conv": torch.tensor([True, True, False, False]),
    }

    apply_masks(model, masks, graph)

    assert model.model[1].cv2.conv.out_channels == int(input_mask.sum())
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 4, 8, 8))
    assert output.shape[1] == int(masks["model.2.conv"].sum())


def build_detect_model() -> TinyModel:
    return TinyModel(
        [
            TinyConv(4, 4),
            TinyConv(4, 4),
            TinyConv(4, 4),
            Detect([0, 1, 2]),
        ]
    )


def test_detect_branches_are_pruned() -> None:
    model = build_detect_model()
    graph = DependencyGraph(model)
    masks = {
        "model.0.conv": make_mask(4, 3),
        "model.1.conv": make_mask(4, 2),
        "model.2.conv": make_mask(4, 1),
    }

    apply_masks(model, masks, graph)

    detect = model.model[3]
    for scale, kept in enumerate((3, 2, 1)):
        assert detect.cv2[scale][0].conv.in_channels == kept
        assert detect.one2one_cv2[scale][0].conv.in_channels == kept

        depthwise = detect.cv3[scale][0][0]
        assert depthwise.conv.in_channels == kept
        assert depthwise.conv.out_channels == kept
        assert depthwise.conv.groups == kept
        assert depthwise.bn.num_features == kept
        assert detect.cv3[scale][0][1].conv.in_channels == kept

        one2one_depthwise = detect.one2one_cv3[scale][0][0]
        assert one2one_depthwise.conv.in_channels == kept
        assert one2one_depthwise.conv.out_channels == kept
        assert one2one_depthwise.conv.groups == kept
        assert one2one_depthwise.bn.num_features == kept
        assert detect.one2one_cv3[scale][0][1].conv.in_channels == kept
