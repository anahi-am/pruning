import pytest
from ultralytics.utils import YAML

from src.finetuning import (
    EXPECTED_CLASSES,
    DatasetLayout,
    build_dataset_config,
    find_label_dir,
    find_split_dir,
    resolve_dataset_layout,
    validate_labels,
)


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_split_dir_prefers_root_images(tmp_path):
    _write(tmp_path / "images" / "train2017" / "a.jpg")

    assert find_split_dir(tmp_path, "train2017") == tmp_path / "images" / "train2017"


def test_find_split_dir_supports_direct_name(tmp_path):
    _write(tmp_path / "train2017" / "a.jpg")

    assert find_split_dir(tmp_path, "train2017") == tmp_path / "train2017"


def test_find_split_dir_returns_none_when_missing(tmp_path):
    assert find_split_dir(tmp_path, "train2017") is None


def test_find_label_dir_supports_labels_folder(tmp_path):
    _write(tmp_path / "labels" / "train2017" / "a.txt")

    assert find_label_dir(tmp_path, "train2017") == tmp_path / "labels" / "train2017"


def test_find_label_dir_supports_split_labels_folder(tmp_path):
    _write(tmp_path / "train" / "labels" / "a.txt")

    assert find_label_dir(tmp_path, "train") == tmp_path / "train" / "labels"


def test_find_label_dir_falls_back_to_nested_split_labels(tmp_path):
    _write(tmp_path / "nested" / "train" / "labels" / "a.txt")

    assert find_label_dir(tmp_path, "train") == tmp_path / "nested" / "train" / "labels"


def test_find_label_dir_returns_none_when_missing(tmp_path):
    assert find_label_dir(tmp_path, "train2017") is None


def test_validate_labels_reports_class_range(tmp_path):
    _write(tmp_path / "a.txt", "0 0.5 0.5 0.1 0.1\n79 0.5 0.5 0.2 0.2\n")

    assert validate_labels(tmp_path, "train") == (0, 79)


def test_validate_labels_ignores_blank_lines(tmp_path):
    _write(tmp_path / "a.txt", "\n0 0.5 0.5 0.1 0.1\n\n")

    assert validate_labels(tmp_path, "train") == (0, 0)


def test_validate_labels_skips_empty_files(tmp_path):
    _write(tmp_path / "empty.txt", "   \n\n")
    _write(tmp_path / "a.txt", "0 0.5 0.5 0.1 0.1\n")

    assert validate_labels(tmp_path, "train") == (0, 0)


def test_validate_labels_rejects_wrong_field_count(tmp_path):
    _write(tmp_path / "a.txt", "0 0.5 0.5 0.1\n")

    with pytest.raises(ValueError, match="expected 5"):
        validate_labels(tmp_path, "train")


def test_validate_labels_rejects_out_of_range_class(tmp_path):
    _write(tmp_path / "a.txt", f"{EXPECTED_CLASSES} 0.5 0.5 0.1 0.1\n")

    with pytest.raises(ValueError, match="class ids range"):
        validate_labels(tmp_path, "train")


def test_validate_labels_rejects_empty_sample(tmp_path):
    _write(tmp_path / "a.txt", "\n\n")

    with pytest.raises(ValueError, match="No annotations"):
        validate_labels(tmp_path, "train")


def test_validate_labels_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="No label files"):
        validate_labels(tmp_path, "train")


def test_resolve_dataset_layout_uses_direct_layout(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        _write(root / split / "images" / "a.jpg")
        _write(root / split / "labels" / "a.txt", "0 0.5 0.5 0.1 0.1\n")

    layout = resolve_dataset_layout(root, tmp_path / "work")

    assert layout == DatasetLayout(
        data_root=root,
        train_entry="train/images",
        val_entry="val/images",
        label_dirs={
            "train": root / "train" / "labels",
            "val": root / "val" / "labels",
        },
        uses_input=True,
    )
    assert not (tmp_path / "work").exists()


def test_resolve_dataset_layout_uses_existing_labels(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train2017", "val2017"):
        _write(root / "images" / split / "a.jpg")
        _write(root / "labels" / split / "a.txt", "0 0.5 0.5 0.1 0.1\n")

    layout = resolve_dataset_layout(root, tmp_path / "work")

    assert layout == DatasetLayout(
        data_root=root,
        train_entry="images/train2017",
        val_entry="images/val2017",
        label_dirs={
            "train": root / "labels" / "train2017",
            "val": root / "labels" / "val2017",
        },
        uses_input=True,
    )
    assert not (tmp_path / "work").exists()


def test_resolve_dataset_layout_rejects_missing_images(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="Could not find images"):
        resolve_dataset_layout(root, tmp_path / "work")


def test_build_dataset_config_writes_train_and_val(tmp_path):
    output = tmp_path / "coco_finetune.yaml"

    config = build_dataset_config(tmp_path, "train/images", "val/images", output)

    assert output.exists()
    assert config["path"] == str(tmp_path)
    assert config["train"] == "train/images"
    assert config["val"] == "val/images"
    assert "test" not in config
    assert "download" not in config
    assert len(config["names"]) == EXPECTED_CLASSES

    saved = YAML.load(output)
    assert saved["train"] == "train/images"
    assert saved["val"] == "val/images"
