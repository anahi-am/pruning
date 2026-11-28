import pytest
import torch

from src.scoring import (
    bn_gamma_scores,
    bn_l1_scores,
    compute_scores,
    magnitude_scores,
)
from tests.helpers import FakeYOLO, TinyConv, TinyModel


def build_model() -> TinyModel:
    return TinyModel([TinyConv(4, 8)])


def test_bn_gamma_scores_use_bn_weight() -> None:
    model = build_model()
    conv = model.model[0]
    with torch.no_grad():
        conv.conv.weight.fill_(10.0)
        conv.bn.weight.copy_(torch.arange(8, dtype=torch.float32))

    scores = bn_gamma_scores(FakeYOLO(model))

    assert torch.equal(scores["model.0.conv"], torch.arange(8, dtype=torch.float32))


def test_bn_gamma_falls_back_to_l1_without_bn() -> None:
    model = TinyModel([torch.nn.Conv2d(4, 3, 1)])
    scores = bn_gamma_scores(FakeYOLO(model))
    expected = model.model[0].weight.data.abs().sum(dim=(1, 2, 3))

    assert torch.equal(scores["model.0"], expected)


def test_bn_l1_scores_multiply_gamma_and_l1() -> None:
    model = build_model()
    conv = model.model[0]
    with torch.no_grad():
        conv.conv.weight.fill_(2.0)
        conv.bn.weight.copy_(torch.arange(8, dtype=torch.float32))

    scores = bn_l1_scores(FakeYOLO(model))

    l1 = conv.conv.weight.data.abs().sum(dim=(1, 2, 3))
    expected = torch.arange(8, dtype=torch.float32) * l1
    assert torch.allclose(scores["model.0.conv"], expected)


def test_bn_l1_falls_back_to_l1_without_bn() -> None:
    model = TinyModel([torch.nn.Conv2d(4, 3, 1)])
    scores = bn_l1_scores(FakeYOLO(model))
    expected = model.model[0].weight.data.abs().sum(dim=(1, 2, 3))

    assert torch.equal(scores["model.0"], expected)


def test_compute_scores_dispatch() -> None:
    model = FakeYOLO(build_model())

    assert set(compute_scores(model, "l1")) == set(magnitude_scores(model))
    assert set(compute_scores(model, "bn")) == set(bn_gamma_scores(model))
    assert set(compute_scores(model, "bn_l1")) == set(bn_l1_scores(model))


def test_compute_scores_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        compute_scores(FakeYOLO(build_model()), "nope")
