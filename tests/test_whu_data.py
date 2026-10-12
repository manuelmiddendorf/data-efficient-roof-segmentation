from pathlib import Path
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from roofseg.whu_data import (
    WHUDataset,
    build_whu_splits,
    load_whu_binary_target,
    validate_whu_roles,
)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = array if array.ndim == 3 else array[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=bands.shape[2],
            height=bands.shape[1],
            count=bands.shape[0],
            dtype=bands.dtype,
        ) as destination:
            destination.write(bands)


def test_whu_pairing_rejects_orphan_label(tmp_path: Path):
    for role in ("train", "val", "test"):
        (tmp_path / role / "image").mkdir(parents=True)
        (tmp_path / role / "label").mkdir()
    _write_tiff(tmp_path / "train/image/0.tif", np.zeros((3, 512, 512), dtype=np.uint8))
    _write_tiff(tmp_path / "train/label/1.tif", np.zeros((512, 512), dtype=np.uint8))

    with pytest.raises(ValueError, match="pairing mismatch"):
        WHUDataset(tmp_path).discover(expected_counts=None)


def test_whu_discovery_is_role_and_numeric_ordered(tmp_path: Path):
    names = {
        "train": ["10.tif", "2.tif"],
        "val": ["val_1.tif"],
        "test": ["2_0.tif", "0.tif"],
    }
    for role, role_names in names.items():
        for directory in ("image", "label"):
            root = tmp_path / role / directory
            root.mkdir(parents=True, exist_ok=True)
            for name in reversed(role_names):
                (root / name).touch()

    samples = WHUDataset(tmp_path).discover(expected_counts=None)

    assert [sample.sample_id for sample in samples] == [
        "train/2", "train/10", "val/val_1", "test/0", "test/2_0"
    ]


def test_whu_binary_target_uses_zero_and_one():
    mask = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(
        load_whu_binary_target(mask), [[True, False, False, True]]
    )


def test_whu_binary_target_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unexpected WHU mask values"):
        load_whu_binary_target(np.array([[0, 255]], dtype=np.uint8))


def test_whu_future_subset_is_deterministic_and_train_only():
    samples = [
        {"id": f"train/{index}", "role": "train"} for index in range(120)
    ] + [{"id": "val/val_0", "role": "val"}, {"id": "test/0", "role": "test"}]
    manifest = {
        "samples": samples,
        "audit": {
            "cross_role_duplicate_image_groups": [],
            "cross_role_duplicate_pair_groups": [],
        },
    }

    first = build_whu_splits(manifest, subset_seed=17)
    second = build_whu_splits(manifest, subset_seed=17)

    assert first == second
    subset = first["future_pilot_subset"]["ids"]
    assert len(subset) == len(set(subset)) == 100
    assert set(subset) <= set(first["roles"]["train"])
    assert set(subset).isdisjoint(first["roles"]["val"] + first["roles"]["test"])


def test_whu_role_validation_rejects_overlap():
    roles = {"train": ["train/0"], "val": ["train/0"], "test": ["test/0"]}
    with pytest.raises(ValueError, match="roles train/val overlap"):
        validate_whu_roles(roles, ["train/0", "test/0"])
