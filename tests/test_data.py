from pathlib import Path

import numpy as np
import pytest

from roofseg.data import RIDDataset, load_binary_target


def test_pairing_rejects_orphan_mask(tmp_path: Path):
    root = tmp_path / "RID_dataset"
    (root / "images_roof_centered_geotiff").mkdir(parents=True)
    (root / "masks_segments_reviewed").mkdir()
    (root / "images_roof_centered_geotiff/1.tif").touch()
    (root / "masks_segments_reviewed/2.png").touch()

    with pytest.raises(ValueError, match="pairing mismatch"):
        RIDDataset(tmp_path).discover()


def test_binary_target_uses_provider_background_code():
    mask = np.array([[0, 8, 16, 17]], dtype=np.uint8)
    np.testing.assert_array_equal(load_binary_target(mask), [[True, True, True, False]])


def test_binary_target_rejects_unknown_codes():
    with pytest.raises(ValueError, match="Unknown RID"):
        load_binary_target(np.array([[18]], dtype=np.uint8))
