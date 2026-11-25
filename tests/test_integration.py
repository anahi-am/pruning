import pytest
import torch
from ultralytics import YOLO

from src.apply_masks import apply_masks
from src.dependency_graph import DependencyGraph
from src.mask_generator import generate_pruning_masks
from src.prune_model import prune_yolo_model

IMAGE_SIZE = 640
RATIO = 0.3


def _forward(model: YOLO) -> None:
    model.model.eval()
    with torch.no_grad():
        model.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))


@pytest.mark.integration
def test_prune_real_model_forward() -> None:
    model = YOLO("yolo26n.pt")
    graph = DependencyGraph(model.model)
    masks = generate_pruning_masks(model, RATIO, graph)

    before = sum(p.numel() for p in model.model.parameters())
    apply_masks(model.model, masks, graph)
    after = sum(p.numel() for p in model.model.parameters())

    assert after < before
    _forward(model)


@pytest.mark.integration
def test_pruned_model_saves_and_reloads(tmp_path) -> None:
    output_path = tmp_path / "pruned.pt"

    prune_yolo_model(output_path=str(output_path))
    assert output_path.exists()

    reloaded = YOLO(str(output_path))
    _forward(reloaded)


@pytest.mark.integration
def test_dry_run_does_not_write(tmp_path) -> None:
    output_path = tmp_path / "pruned.pt"

    prune_yolo_model(output_path=str(output_path), dry_run=True)
    assert not output_path.exists()
