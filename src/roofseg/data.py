"""RID pairing, raster inspection and binary roof-target construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds


ROOF_SEGMENT_CODES = tuple(range(17))
BACKGROUND_CODE = 17
ALLOWED_MASK_CODES = (*ROOF_SEGMENT_CODES, BACKGROUND_CODE)


@dataclass(frozen=True)
class SampleRecord:
    """One roof-centred image/mask pair using the provider's numeric ID."""

    sample_id: str
    image_path: Path
    mask_path: Path


class RIDDataset:
    """Strict reader for the reviewed roof-segment subset of RID release 1.0."""

    def __init__(self, release_root: Path):
        self.release_root = release_root
        self.dataset_root = release_root / "RID_dataset"
        self.image_dir = self.dataset_root / "images_roof_centered_geotiff"
        self.mask_dir = self.dataset_root / "masks_segments_reviewed"

    def discover(self) -> list[SampleRecord]:
        """Pair all GeoTIFF images and reviewed segment masks by numeric stem.

        Raises
        ------
        ValueError
            If names are non-numeric, duplicated or the two ID sets differ.
        """
        images = _indexed_files(self.image_dir, ".tif")
        masks = _indexed_files(self.mask_dir, ".png")
        if set(images) != set(masks):
            missing_masks = sorted(set(images) - set(masks), key=int)
            missing_images = sorted(set(masks) - set(images), key=int)
            raise ValueError(
                f"RID pairing mismatch; missing masks={missing_masks[:8]}, "
                f"missing images={missing_images[:8]}."
            )
        return [SampleRecord(sample_id, images[sample_id], masks[sample_id])
                for sample_id in sorted(images, key=int)]


def _indexed_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required RID directory is missing: {directory}")
    indexed: dict[str, Path] = {}
    unexpected = [path.name for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() != suffix]
    if unexpected:
        raise ValueError(f"Unexpected files in {directory.name}: {unexpected[:8]}")
    for path in directory.glob(f"*{suffix}"):
        if not path.stem.isdigit() or path.stem in indexed:
            raise ValueError(f"Invalid or duplicate RID sample name: {path.name}")
        indexed[path.stem] = path
    return indexed


def read_image(path: Path) -> np.ndarray:
    """Read one RID GeoTIFF as uint8 RGB with shape ``(512, 512, 3)``."""
    with rasterio.open(path) as source:
        array = source.read()
    if array.shape != (3, 512, 512) or array.dtype != np.uint8:
        raise ValueError(f"Unexpected RID image contract in {path}: {array.shape}, {array.dtype}")
    return np.moveaxis(array, 0, -1)


def read_mask(path: Path) -> np.ndarray:
    """Read one reviewed roof-segment mask as uint8 ``(512, 512)``."""
    with Image.open(path) as image:
        array = np.asarray(image)
        mode = image.mode
    if mode != "L" or array.shape != (512, 512) or array.dtype != np.uint8:
        raise ValueError(f"Unexpected RID mask contract in {path}: {mode}, {array.shape}, {array.dtype}")
    validate_mask_codes(array)
    return array


def validate_mask_codes(mask: np.ndarray) -> tuple[int, ...]:
    """Return observed codes after requiring provider-defined values 0--17."""
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("RID segment masks must be two-dimensional integer arrays.")
    values = tuple(int(value) for value in np.unique(mask))
    unexpected = sorted(set(values) - set(ALLOWED_MASK_CODES))
    if unexpected:
        raise ValueError(f"Unknown RID roof-segment codes: {unexpected}")
    return values


def load_binary_target(mask: np.ndarray) -> np.ndarray:
    """Map reviewed roof-segment codes to a binary roof target.

    Codes 0--15 are the azimuth classes, code 16 is flat roof and code 17 is
    background. This ordering is verified against the provider's mask-generation
    code. No unknown/unlabelled class exists in this mask product; this encoding
    does not prove annotation completeness.

    Returns
    -------
    ndarray
        Boolean array with the same height and width as ``mask``.
    """
    validate_mask_codes(mask)
    return mask != BACKGROUND_CODE


def raster_metadata(path: Path) -> dict:
    """Read dimensions, channels, CRS, affine transform and WGS84 bounds."""
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"RID image lacks CRS: {path}")
        bounds_wgs84 = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
        return {
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtype": list(source.dtypes),
            "crs": source.crs.to_string(),
            "transform": [float(value) for value in source.transform[:6]],
            "bounds_native": [float(value) for value in source.bounds],
            "bounds_wgs84": [float(value) for value in bounds_wgs84],
            "nodata": source.nodata,
            "valid_pixels": int(np.count_nonzero(source.dataset_mask())),
        }


def parse_split_file(path: Path) -> list[str]:
    """Read provider split filenames as unique numeric IDs in file order."""
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate entries in split file: {path}")
    if any(Path(name).suffix.lower() != ".png" or not Path(name).stem.isdigit() for name in names):
        raise ValueError(f"Unexpected entry in split file: {path}")
    return [Path(name).stem for name in names]


def check_role_partition(roles: dict[str, Iterable[str]], expected_ids: Iterable[str]) -> None:
    """Require disjoint roles that contain every expected sample exactly once."""
    sets = {name: set(ids) for name, ids in roles.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Roles {left}/{right} overlap: {sorted(overlap, key=int)[:8]}")
    combined = set().union(*sets.values())
    expected = set(expected_ids)
    if combined != expected:
        raise ValueError(
            f"Role partition mismatch; missing={sorted(expected - combined, key=int)[:8]}, "
            f"unexpected={sorted(combined - expected, key=int)[:8]}."
        )
