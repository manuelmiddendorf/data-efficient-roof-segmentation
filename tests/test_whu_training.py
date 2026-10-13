import numpy as np
import pandas as pd

from roofseg.training import reference_group_metrics
from roofseg.training_data import paired_schedule
from roofseg.whu_pilot_analysis import (
    choose_whu_validation_examples, whu_initialization_effects,
    whu_learning_curve_comparisons,
)
from roofseg.whu_pilot import build_whu_pilot_config, whu_run_name, whu_subset_record


def _splits():
    splits = {
        "split_name": "whu_fixture",
        "roles": {"train": [f"train/{i}" for i in range(600)], "val": ["val/1", "val/2"], "test": ["test/1"]},
        "future_pilot_subset": {"size": 100, "seed": 17, "ids": [f"train/{i}" for i in range(100)]},
        "additional_pilot_subsets": {"repetitions": {
            "2": {"size": 100, "seed": 29, "ids": [f"train/{i}" for i in range(100, 200)]},
            "3": {"size": 100, "seed": 43, "ids": [f"train/{i}" for i in range(200, 300)]},
        }},
    }
    splits["learning_curve_subsets"] = {"repetitions": {
        "1": {"subset_seed": 17, "subsets": {
            "25": [f"train/{i}" for i in range(25)],
            "100": splits["future_pilot_subset"]["ids"],
            "500": [f"train/{i}" for i in range(500)],
        }},
        "2": {"subset_seed": 29, "subsets": {
            "25": [f"train/{i}" for i in range(100, 125)],
            "100": splits["additional_pilot_subsets"]["repetitions"]["2"]["ids"],
            "500": [f"train/{i}" for i in range(100, 600)],
        }},
    }}
    return splits


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


def test_whu_repetitions_change_only_subset_identity_and_run_name():
    splits = _splits()
    configs = [build_whu_pilot_config(splits, "random", repetition) for repetition in (1, 2, 3)]
    assert [config["seeds"]["subset"] for config in configs] == [17, 29, 43]
    assert len({config["run_name"] for config in configs}) == 3
    assert whu_subset_record(splits, 1) is splits["future_pilot_subset"]
    for config in configs[1:]:
        reference = configs[0]
        differing = {
            key for key in reference
            if reference[key] != config[key]
        }
        assert differing == {"run_name", "data", "training_ids", "seeds"}
        assert {key for key in reference["data"] if reference["data"][key] != config["data"][key]} == {"training_subset"}
        assert {key for key in reference["seeds"] if reference["seeds"][key] != config["seeds"][key]} == {"subset"}


def test_whu_learning_curve_configs_are_nested_paired_and_uniquely_named():
    splits = _splits()
    for repetition in (1, 2):
        subsets = [
            build_whu_pilot_config(splits, "random", repetition, size)
            for size in (25, 100, 500)
        ]
        assert set(subsets[0]["training_ids"]) < set(subsets[1]["training_ids"])
        assert set(subsets[1]["training_ids"]) < set(subsets[2]["training_ids"])
        for size, random in zip((25, 100, 500), subsets):
            imagenet = build_whu_pilot_config(splits, "imagenet", repetition, size)
            assert random["training_ids"] == imagenet["training_ids"]
            assert random["seeds"] == imagenet["seeds"]
            assert random["run_name"] != imagenet["run_name"]
    names = {
        build_whu_pilot_config(splits, initialization, repetition, size)["run_name"]
        for repetition in (1, 2) for size in (25, 100, 500)
        for initialization in ("random", "imagenet")
    }
    assert len(names) == 12


def test_whu_n25_schedule_retains_incomplete_batches():
    config = build_whu_pilot_config(_splits(), "random", 1, 25)
    schedule, record = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    assert record["incomplete_batches_retained"]
    assert any(len(batch_ids) == 1 for batch_ids, _ in schedule)


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


def test_common_examples_use_paired_reference_defined_rules():
    rows = []
    for initialization, ious, fps in (
        ("random", [0.2, 0.9, 1.0, 0.0], [0, 0, 0, 8]),
        ("imagenet", [0.4, 0.8, 1.0, 1.0], [0, 0, 0, 0]),
    ):
        for sample_id, positive, iou, fp in zip(
            ("positive-low", "positive-high", "empty-low", "empty-high"),
            (True, True, False, False), ious, fps,
        ):
            rows.append({
                "initialization": initialization, "subset_seed": 17,
                "split": "validation", "id": sample_id,
                "tp": 1 if positive else 0, "fn": 0, "fp": fp, "iou": iou,
            })
    selected = choose_whu_validation_examples(pd.DataFrame(rows))
    assert selected.id.tolist() == [
        "positive-low", "positive-high", "empty-low", "empty-high"
    ]


def test_whu_initialization_effects_are_paired_by_subset_seed():
    results = pd.DataFrame([
        {"subset_seed": 17, "initialization": "random", "best_validation_iou": 0.7,
         "last_validation_iou": 0.6},
        {"subset_seed": 17, "initialization": "imagenet", "best_validation_iou": 0.8,
         "last_validation_iou": 0.75},
        {"subset_seed": 29, "initialization": "random", "best_validation_iou": 0.6,
         "last_validation_iou": 0.55},
        {"subset_seed": 29, "initialization": "imagenet", "best_validation_iou": 0.75,
         "last_validation_iou": 0.65},
    ])
    effects, summary = whu_initialization_effects(results)
    assert np.allclose(effects.imagenet_minus_random_best_iou, [0.1, 0.15])
    assert np.isclose(summary.iloc[0].mean_imagenet_minus_random_best_iou, 0.125)
    assert np.isclose(summary.iloc[0].sample_sd_imagenet_minus_random_best_iou,
                      np.std([0.1, 0.15], ddof=1))


def test_whu_learning_curve_comparisons_pair_sizes_and_compute_gains():
    rows = []
    for size, random_iou, imagenet_iou in ((25, 0.5, 0.6), (100, 0.6, 0.72), (500, 0.7, 0.8)):
        for initialization, best in (("random", random_iou), ("imagenet", imagenet_iou)):
            rows.append({
                "subset_repetition": 1, "subset_seed": 17, "training_size": size,
                "initialization": initialization, "best_validation_iou": best,
                "last_validation_iou": best - 0.01,
            })
    splits = {"learning_curve_subsets": {"pairwise_id_overlaps": [
        {"training_size": 25, "first_seed": 17, "second_seed": 29, "shared_ids": 0}
    ]}}
    effects, gains, summary, overlaps = whu_learning_curve_comparisons(
        pd.DataFrame(rows), splits
    )
    assert np.allclose(effects.imagenet_minus_random_best_iou, [0.1, 0.12, 0.1])
    random_gains = gains[gains.initialization == "random"]
    assert np.allclose(random_gains.best_iou_gain, [0.1, 0.1])
    assert summary.best_iou_sample_sd.isna().all()
    assert overlaps.shared_ids.tolist() == [0]
