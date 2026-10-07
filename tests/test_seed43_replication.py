from pathlib import Path

from roofseg.boundary_weighted import boundary_run_directory, build_boundary_config
from roofseg.optimization import build_late_lr_drop_config, late_lr_drop_run_directory


def test_seed43_reference_and_boundary_pair_are_distinct_and_paired(tmp_path: Path):
    ids25 = [str(index) for index in range(25)]
    ids100 = [str(index) for index in range(100)]
    ids500 = [str(index) for index in range(500)]
    splits = {
        "split_name": "geographic_v2",
        "roles": {
            "training": ids500,
            "validation": ["600"],
            "test": ["700"],
        },
        "training_subsets": {"repetitions": {
            "1": {"seed": 17, "subsets": {
                "25": ids25, "100": ids100, "500": ids500,
            }},
            "3": {"seed": 43, "subsets": {
                "25": ids25, "100": ids100, "500": ids500,
            }},
        }},
    }
    references = {
        (size, initialization): build_late_lr_drop_config(
            splits, initialization, 3, size
        )
        for size in (25, 100, 500)
        for initialization in ("random", "imagenet")
    }
    assert set(references[(25, "random")]["training_ids"]) < set(
        references[(100, "random")]["training_ids"]
    ) < set(references[(500, "random")]["training_ids"])
    for size in (25, 100, 500):
        assert references[(size, "random")]["training_ids"] == (
            references[(size, "imagenet")]["training_ids"]
        )
        assert references[(size, "random")]["seeds"]["subset"] == 43

    coefficient = 0.041
    boundary_pair = {
        initialization: build_boundary_config(
            splits, initialization, 3, "equal_per_image", coefficient
        )
        for initialization in ("random", "imagenet")
    }
    assert boundary_pair["random"]["training_ids"] == references[(25, "random")]["training_ids"]
    assert boundary_pair["imagenet"]["training_ids"] == references[(25, "imagenet")]["training_ids"]
    assert {
        config["boundary_weighted_bce"]["mean_training_band_fraction"]
        for config in boundary_pair.values()
    } == {coefficient}
    directories = {
        late_lr_drop_run_directory(tmp_path, splits, initialization, 3, size)
        for size in (25, 100, 500)
        for initialization in ("random", "imagenet")
    } | {
        boundary_run_directory(tmp_path, splits, initialization, 3, "equal_per_image")
        for initialization in ("random", "imagenet")
    }
    assert len(directories) == 8
