import numpy as np

from roofseg.training import reference_group_metrics
from roofseg.whu_pilot import build_whu_pilot_config, whu_run_name


def _splits():
    return {
        "split_name": "whu_fixture",
        "roles": {"train": ["train/1", "train/2"], "val": ["val/1", "val/2"], "test": ["test/1"]},
        "future_pilot_subset": {"size": 100, "seed": 17, "ids": [f"train/{i}" for i in range(100)]},
    }


def test_whu_pilot_config_uses_saved_subset_and_provider_validation():
    splits = _splits()
    random = build_whu_pilot_config(splits, "random")
    imagenet = build_whu_pilot_config(splits, "imagenet")
    assert random["training_ids"] == splits["future_pilot_subset"]["ids"]
    assert random["training_ids"] == imagenet["training_ids"]
    assert random["validation_ids"] == splits["roles"]["val"]
    assert not (set(random["training_ids"]) | set(random["validation_ids"])) & set(splits["roles"]["test"])
    assert random["seeds"] == imagenet["seeds"]
    assert random["scheduler"] == imagenet["scheduler"]
    assert whu_run_name("random") != whu_run_name("imagenet")


def test_reference_group_metrics_keep_empty_images_separate():
    rows = [
        {"tp": 2, "fp": 1, "fn": 1, "iou": 0.5, "dice": 2 / 3, "average_precision": 0.75},
        {"tp": 0, "fp": 2, "fn": 0, "iou": 0.0, "dice": 0.0, "average_precision": None},
        {"tp": 0, "fp": 0, "fn": 0, "iou": 1.0, "dice": 1.0, "average_precision": None},
    ]
    grouped = reference_group_metrics(rows, pixels_per_image=4)
    assert grouped["positive_reference"]["image_count"] == 1
    assert grouped["positive_reference"]["mean_average_precision"] == 0.75
    assert grouped["empty_reference"]["image_count"] == 2
    assert np.isclose(grouped["empty_reference"]["mean_predicted_building_fraction"], 0.25)
    assert grouped["empty_reference"]["fraction_with_any_predicted_building"] == 0.5
