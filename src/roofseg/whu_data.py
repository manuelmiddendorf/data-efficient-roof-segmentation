"""WHU Aerial raster pairing, validation, audit, and report figures."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from .integrity import sha256_file


SOURCE_PAGE = "https://gpcv.whu.edu.cn/data/building_dataset.html"
ARCHIVE_URL = (
    "https://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20"
    "and%20raster%20labels.zip"
)
ARCHIVE_NAME = "whu_aerial_0.3m_raster.zip"
DATASET_DIRECTORY = "3. The cropped image tiles and raster labels"
ROLE_ORDER = ("train", "val", "test")
EXPECTED_ROLE_COUNTS = {"train": 4736, "val": 1036, "test": 2416}
DOCUMENTED_TILE_BLOCKS = {
    "train": [{"width_tiles": 64, "height_tiles": 74}],
    "val": [{"width_tiles": 14, "height_tiles": 74}],
    "test": [
        {"width_tiles": 17, "height_tiles": 42},
        {"width_tiles": 23, "height_tiles": 74},
    ],
}


@dataclass(frozen=True)
class WHUSample:
    """One provider-role-specific WHU image/label pair."""

    sample_id: str
    role: str
    provider_name: str
    provider_block: int
    provider_index: int
    image_path: Path
    mask_path: Path


class WHUDataset:
    """Strict reader for the prepared 0.3 m WHU Aerial raster release."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    def discover(
        self, expected_counts: dict[str, int] | None = EXPECTED_ROLE_COUNTS
    ) -> list[WHUSample]:
        """Pair TIFF images and labels within each provider role.

        Parameters
        ----------
        expected_counts:
            Required role counts. Pass ``None`` only for small synthetic tests.

        Returns
        -------
        list of WHUSample
            Pairs sorted by provider role and numeric crop index.
        """
        samples: list[WHUSample] = []
        for role in ROLE_ORDER:
            images = _indexed_tiffs(self.dataset_root / role / "image", role)
            masks = _indexed_tiffs(self.dataset_root / role / "label", role)
            if set(images) != set(masks):
                raise ValueError(
                    f"WHU {role} pairing mismatch; missing labels="
                    f"{sorted(set(images) - set(masks))[:8]}, missing images="
                    f"{sorted(set(masks) - set(images))[:8]}."
                )
            if expected_counts is not None and len(images) != expected_counts[role]:
                raise ValueError(
                    f"Expected {expected_counts[role]} WHU {role} pairs, found {len(images)}."
                )
            for provider_name, indexed_path in sorted(
                images.items(), key=lambda item: item[1][0]
            ):
                (block, index), image_path = indexed_path
                samples.append(
                    WHUSample(
                        sample_id=f"{role}/{provider_name}",
                        role=role,
                        provider_name=provider_name,
                        provider_block=block,
                        provider_index=index,
                        image_path=image_path,
                        mask_path=masks[provider_name][1],
                    )
                )
        return samples


def _indexed_tiffs(directory: Path, role: str) -> dict[str, tuple[tuple[int, int], Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required WHU directory is missing: {directory}")
    files = [path for path in directory.iterdir() if path.is_file()]
    unexpected = [path.name for path in files if path.suffix.lower() not in {".tif", ".tiff"}]
    if unexpected:
        raise ValueError(f"Unexpected files in {directory}: {sorted(unexpected)[:8]}")
    indexed: dict[str, tuple[tuple[int, int], Path]] = {}
    for path in files:
        stem = path.stem
        if role == "val" and stem.startswith("val_"):
            block, numeric = 1, stem[4:]
        elif role == "test" and stem.startswith("2_"):
            block, numeric = 2, stem[2:]
        else:
            block, numeric = 1, stem
        valid_prefix = (
            (role == "val" and stem.startswith("val_"))
            or (role == "test" and (stem.isdigit() or stem.startswith("2_")))
            or (role == "train" and stem.isdigit())
        )
        if not numeric.isdigit() or not valid_prefix:
            raise ValueError(f"Unexpected WHU {role} filename: {path.name}")
        if stem in indexed:
            raise ValueError(f"Duplicate WHU {role} filename: {path.name}")
        indexed[stem] = ((block, int(numeric)), path)
    return indexed


def read_whu_image(path: Path) -> np.ndarray:
    """Read one WHU tile as ``uint8`` RGB with shape ``(512, 512, 3)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            array = source.read()
    if array.shape != (3, 512, 512) or array.dtype != np.uint8:
        raise ValueError(
            f"Unexpected WHU image contract in {path}: {array.shape}, {array.dtype}"
        )
    return np.moveaxis(array, 0, -1)


def read_whu_mask(path: Path) -> np.ndarray:
    """Read one WHU building mask as ``uint8`` with shape ``(512, 512)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            array = source.read()
    if array.shape != (1, 512, 512) or array.dtype != np.uint8:
        raise ValueError(
            f"Unexpected WHU mask contract in {path}: {array.shape}, {array.dtype}"
        )
    mask = array[0]
    values = set(int(value) for value in np.unique(mask))
    if not values <= {0, 1}:
        raise ValueError(f"Unexpected WHU mask values in {path}: {sorted(values)}")
    return mask


def load_whu_binary_target(mask: np.ndarray) -> np.ndarray:
    """Map WHU raster labels to a Boolean building-footprint target.

    The prepared 0.3 m archive uses 0 for annotated building footprint and 1
    for background. It contains no separate invalid or unknown label code.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("WHU masks must be two-dimensional uint8 arrays.")
    values = set(int(value) for value in np.unique(mask))
    if not values <= {0, 1}:
        raise ValueError(f"Unexpected WHU mask values: {sorted(values)}")
    return mask == 0


def _raster_metadata(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            gcps, gcp_crs = source.gcps
            return {
                "driver": source.driver,
                "width": source.width,
                "height": source.height,
                "bands": source.count,
                "dtype": list(source.dtypes),
                "crs": source.crs.to_string() if source.crs else None,
                "transform": [float(value) for value in source.transform[:6]],
                "transform_is_identity": bool(source.transform.is_identity),
                "gcp_count": len(gcps),
                "gcp_crs": gcp_crs.to_string() if gcp_crs else None,
                "nodata": source.nodata,
                "valid_pixels": int(np.count_nonzero(source.dataset_mask())),
            }


def _inspect_sample(sample: WHUSample, project_root: Path) -> dict:
    image = read_whu_image(sample.image_path)
    mask = read_whu_mask(sample.mask_path)
    return {
        "id": sample.sample_id,
        "role": sample.role,
        "provider_name": sample.provider_name,
        "provider_block": sample.provider_block,
        "provider_index": sample.provider_index,
        "image": sample.image_path.relative_to(project_root).as_posix(),
        "mask": sample.mask_path.relative_to(project_root).as_posix(),
        "image_sha256": sha256_file(sample.image_path),
        "mask_sha256": sha256_file(sample.mask_path),
        "image_bytes": sample.image_path.stat().st_size,
        "mask_bytes": sample.mask_path.stat().st_size,
        "image_min": int(image.min()),
        "image_max": int(image.max()),
        "mask_values": [int(value) for value in np.unique(mask)],
        "building_pixels": int(np.count_nonzero(mask == 0)),
    }


def _duplicate_groups(samples: Iterable[dict], keys: tuple[str, ...]) -> list[list[str]]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sample in samples:
        groups[tuple(sample[key] for key in keys)].append(sample["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def _cross_role_groups(groups: Iterable[list[str]]) -> list[list[str]]:
    return [group for group in groups if len({sample_id.split("/", 1)[0] for sample_id in group}) > 1]


def build_whu_manifest(project_root: Path) -> dict:
    """Inspect every official prepared WHU Aerial image/label pair."""
    raw_root = project_root / "data/raw/whu"
    archive_path = raw_root / ARCHIVE_NAME
    dataset_root = raw_root / DATASET_DIRECTORY
    if not archive_path.is_file():
        raise FileNotFoundError(f"Official WHU archive is missing: {archive_path}")
    pairs = WHUDataset(dataset_root).discover()
    with ThreadPoolExecutor(max_workers=8) as executor:
        samples = list(executor.map(lambda pair: _inspect_sample(pair, project_root), pairs))

    image_metadata = [_raster_metadata(pair.image_path) for pair in pairs]
    mask_metadata = [_raster_metadata(pair.mask_path) for pair in pairs]
    image_duplicates = _duplicate_groups(samples, ("image_sha256",))
    mask_duplicates = _duplicate_groups(samples, ("mask_sha256",))
    pair_duplicates = _duplicate_groups(samples, ("image_sha256", "mask_sha256"))
    role_counts = {
        role: sum(sample["role"] == role for sample in samples) for role in ROLE_ORDER
    }
    return {
        "schema_version": 1,
        "dataset": {
            "name": "WHU Building Dataset – Aerial imagery subset",
            "variant": "prepared 0.3 m raster images and labels",
            "source_page": SOURCE_PAGE,
            "archive_url": ARCHIVE_URL,
            "archive": f"data/raw/whu/{ARCHIVE_NAME}",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
            "hash_status": (
                "Locally computed identity only; the provider page does not publish "
                "a trusted checksum for this archive."
            ),
            "downloaded": "2026-10-12",
            "http_last_modified": "2018-07-03T01:58:46Z",
            "paper": (
                "Ji, Wei and Lu (2019), Fully Convolutional Networks for Multisource "
                "Building Extraction from an Open Aerial and Satellite Imagery Data Set"
            ),
            "paper_doi": "10.1109/TGRS.2018.2858817",
            "terms": (
                "The official GPCV index describes the datasets as freely downloadable, "
                "but the dataset page and archive contain no explicit reuse licence. "
                "Redistribution or use beyond this research project requires checking "
                "permission with the provider and the underlying LINZ imagery terms."
            ),
            "attribution": (
                "Cite Ji, Wei and Lu (2019), acknowledge the GPCV Lab at Wuhan "
                "University, and identify New Zealand LINZ as the original imagery source."
            ),
        },
        "label_contract": {
            "product": "prepared raster building labels",
            "building_values": [0],
            "background_values": [1],
            "binary_mapping": "building = mask == 0",
            "mapping_evidence": (
                "Direct inspection of development pairs: all-one labels align with "
                "tiles containing no buildings, while zero-valued regions align with "
                "visible building footprints."
            ),
            "validity": (
                "Every pixel has value 0 or 1; the files provide no distinct invalid or "
                "unknown code and no nodata declaration."
            ),
            "target_scope": (
                "Rasterized building footprints edited from Christchurch building vectors. "
                "This is a building-footprint target, whereas RID masks describe visible "
                "roof segments around roof-centred samples."
            ),
        },
        "audit": {
            "pair_count": len(samples),
            "role_counts": role_counts,
            "provider_page_tile_total": 8189,
            "role_count_total": sum(EXPECTED_ROLE_COUNTS.values()),
            "count_note": (
                "The downloadable role directories contain 8,188 pairs, matching the "
                "provider's displayed role arithmetic (4,736 + 1,036 + 2,416), although "
                "the surrounding page text states 8,189 tiles."
            ),
            "image_shapes_chw": [
                list(shape)
                for shape in sorted(
                    {(item["bands"], item["height"], item["width"]) for item in image_metadata}
                )
            ],
            "mask_shapes_chw": [
                list(shape)
                for shape in sorted(
                    {(item["bands"], item["height"], item["width"]) for item in mask_metadata}
                )
            ],
            "image_dtypes": sorted({dtype for item in image_metadata for dtype in item["dtype"]}),
            "mask_dtypes": sorted({dtype for item in mask_metadata for dtype in item["dtype"]}),
            "image_value_range": [
                min(sample["image_min"] for sample in samples),
                max(sample["image_max"] for sample in samples),
            ],
            "mask_values_observed": sorted(
                {value for sample in samples for value in sample["mask_values"]}
            ),
            "empty_mask_counts": {
                role: sum(
                    sample["role"] == role and sample["building_pixels"] == 0
                    for sample in samples
                )
                for role in ROLE_ORDER
            },
            "all_pixels_valid_by_file_mask": all(
                item["valid_pixels"] == 512 * 512 for item in image_metadata + mask_metadata
            ),
            "crs_values": sorted({item["crs"] for item in image_metadata}, key=str),
            "all_transforms_identity": all(item["transform_is_identity"] for item in image_metadata),
            "all_gcp_counts_zero": all(item["gcp_count"] == 0 for item in image_metadata),
            "all_nodata_unset": all(item["nodata"] is None for item in image_metadata + mask_metadata),
            "documented_ground_sample_distance_m": 0.3,
            "ground_sample_distance_evidence": "provider documentation, not embedded TIFF metadata",
            "exact_duplicate_image_groups": image_duplicates,
            "exact_duplicate_mask_groups": mask_duplicates,
            "exact_duplicate_pair_groups": pair_duplicates,
            "cross_role_duplicate_image_groups": _cross_role_groups(image_duplicates),
            "cross_role_duplicate_pair_groups": _cross_role_groups(pair_duplicates),
        },
        "samples": samples,
    }


def validate_whu_roles(roles: dict[str, Iterable[str]], expected_ids: Iterable[str]) -> None:
    """Require disjoint provider roles containing every manifest ID exactly once."""
    role_sets = {role: set(ids) for role, ids in roles.items()}
    for index, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[index + 1 :]:
            overlap = role_sets[left] & role_sets[right]
            if overlap:
                raise ValueError(f"WHU roles {left}/{right} overlap: {sorted(overlap)[:8]}")
    combined = set().union(*role_sets.values())
    expected = set(expected_ids)
    if combined != expected:
        raise ValueError(
            f"WHU role partition mismatch; missing={sorted(expected - combined)[:8]}, "
            f"unexpected={sorted(combined - expected)[:8]}."
        )


def _uniform_training_subset(training_ids: list[str], seed: int, size: int = 100) -> list[str]:
    """Draw one sorted-position uniform subset with an isolated NumPy generator."""
    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(len(training_ids), size=size, replace=False)
    return [training_ids[int(position)] for position in sorted(selected_positions)]


def _nested_whu_learning_curve_subsets(
    training_ids: list[str], existing_100_ids: list[str], subset_seed: int
) -> dict:
    """Extend one historical n=100 selection to fixed nested n=25/100/500 sets."""
    position = {sample_id: index for index, sample_id in enumerate(training_ids)}
    if len(existing_100_ids) != 100 or len(set(existing_100_ids)) != 100:
        raise ValueError("WHU learning curves require 100 unique historical IDs.")
    if not set(existing_100_ids) <= set(training_ids):
        raise ValueError("Historical WHU subset contains IDs outside the training role.")

    small_seed = subset_seed * 1000 + 25
    extension_seed = subset_seed * 1000 + 500
    small_rng = np.random.default_rng(small_seed)
    small_positions = small_rng.choice(100, size=25, replace=False)
    subset_25 = sorted(
        (existing_100_ids[int(index)] for index in small_positions), key=position.get
    )

    existing_set = set(existing_100_ids)
    remaining_ids = [sample_id for sample_id in training_ids if sample_id not in existing_set]
    extension_rng = np.random.default_rng(extension_seed)
    extension_positions = extension_rng.choice(len(remaining_ids), size=400, replace=False)
    extension = [remaining_ids[int(index)] for index in extension_positions]
    subset_500 = sorted([*existing_100_ids, *extension], key=position.get)
    return {
        "subset_seed": subset_seed,
        "derived_seeds": {"n25_from_n100": small_seed, "n500_extension": extension_seed},
        "subsets": {"25": subset_25, "100": list(existing_100_ids), "500": subset_500},
    }


def build_whu_splits(manifest: dict, subset_seed: int = 17) -> dict:
    """Record provider roles and three label-independent n=100 subset repetitions."""
    samples = manifest["samples"]
    roles = {
        role: [sample["id"] for sample in samples if sample["role"] == role]
        for role in ROLE_ORDER
    }
    validate_whu_roles(roles, [sample["id"] for sample in samples])
    pilot_ids = _uniform_training_subset(roles["train"], subset_seed)
    additional = {
        str(repetition): {
            "size": 100,
            "seed": seed,
            "ids": _uniform_training_subset(roles["train"], seed),
        }
        for repetition, seed in ((2, 29), (3, 43))
    }
    all_subsets = {"1": {"seed": subset_seed, "ids": pilot_ids}, **additional}
    overlaps = []
    for first, second in (("1", "2"), ("1", "3"), ("2", "3")):
        overlaps.append({
            "first_repetition": int(first),
            "first_seed": all_subsets[first]["seed"],
            "second_repetition": int(second),
            "second_seed": all_subsets[second]["seed"],
            "shared_ids": len(set(all_subsets[first]["ids"]) & set(all_subsets[second]["ids"])),
        })
    learning_curve_repetitions = {
        repetition: _nested_whu_learning_curve_subsets(
            roles["train"], all_subsets[repetition]["ids"], all_subsets[repetition]["seed"]
        )
        for repetition in ("1", "2")
    }
    learning_curve_overlaps = [
        {
            "training_size": size,
            "first_seed": learning_curve_repetitions["1"]["subset_seed"],
            "second_seed": learning_curve_repetitions["2"]["subset_seed"],
            "shared_ids": len(
                set(learning_curve_repetitions["1"]["subsets"][str(size)])
                & set(learning_curve_repetitions["2"]["subsets"][str(size)])
            ),
        }
        for size in (25, 100, 500)
    ]
    return {
        "schema_version": 1,
        "dataset_manifest": "data/metadata/whu/data_manifest.json",
        "split_name": "whu_aerial_provider_split_v1",
        "roles": roles,
        "counts": {role: len(ids) for role, ids in roles.items()},
        "spatial_audit": {
            "provider_documented_tile_blocks": DOCUMENTED_TILE_BLOCKS,
            "provider_statement": (
                "The source page shows separate training, evaluation and two test areas "
                "and gives each area's tile-grid dimensions."
            ),
            "embedded_georeferencing": False,
            "per_tile_coordinates_available": False,
            "numeric_order_used_as_coordinates": False,
            "exact_cross_role_image_duplicates": len(
                manifest["audit"]["cross_role_duplicate_image_groups"]
            ),
            "exact_cross_role_pair_duplicates": len(
                manifest["audit"]["cross_role_duplicate_pair_groups"]
            ),
            "visible_area_overlap_status": (
                "Not independently computable from the prepared tiles: TIFFs contain no "
                "CRS, geotransform or GCPs, and the provider does not publish an explicit "
                "tile-index-to-coordinate table in this archive. The documented region "
                "layout supports geographic separation but cannot verify every footprint."
            ),
        },
        "future_pilot_subset": {
            "size": 100,
            "seed": subset_seed,
            "method": (
                "Uniform sampling without replacement from the provider training role, "
                "using an isolated NumPy generator and no labels."
            ),
            "ids": pilot_ids,
            "spatial_maximin_status": (
                "Not defensible from the prepared release because physical tile "
                "coordinates cannot be reconstructed reliably."
            ),
            "test_used": False,
        },
        "additional_pilot_subsets": {
            "method": (
                "Uniform sampling without replacement from the same ordered provider "
                "training-role list, using an isolated NumPy generator and no labels."
            ),
            "repetitions": additional,
            "pairwise_id_overlaps": overlaps,
            "test_used": False,
        },
        "learning_curve_subsets": {
            "sizes": [25, 100, 500],
            "repetitions": learning_curve_repetitions,
            "method": (
                "The historical n=100 selection is not the prefix set of a NumPy "
                "permutation generated with its original seed. Following the predeclared "
                "fallback, n=25 is sampled without replacement from n=100 and n=500 adds "
                "400 samples drawn without replacement from the remaining provider training "
                "role. Source lists use provider numeric order; isolated derived seeds are "
                "subset_seed * 1000 + 25 and subset_seed * 1000 + 500. Labels, masks and "
                "model outputs are not used."
            ),
            "original_n100_is_same_seed_permutation_prefix": {"17": False, "29": False},
            "pairwise_id_overlaps": learning_curve_overlaps,
            "test_used": False,
        },
    }


def build_whu_report(manifest: dict, splits: dict) -> dict:
    """Return compact notebook-facing values from the canonical WHU records."""
    role_summaries = {}
    for role in ROLE_ORDER:
        fractions = np.asarray(
            [
                sample["building_pixels"] / (512 * 512)
                for sample in manifest["samples"]
                if sample["role"] == role
            ]
        )
        role_summaries[role] = {
            "count": len(fractions),
            "empty_masks": int(np.count_nonzero(fractions == 0)),
            "minimum": float(fractions.min()),
            "median": float(np.median(fractions)),
            "mean": float(fractions.mean()),
            "maximum": float(fractions.max()),
        }
    return {
        "dataset": manifest["dataset"],
        "label_contract": manifest["label_contract"],
        "audit": {key: value for key, value in manifest["audit"].items() if not key.endswith("_groups")},
        "duplicate_group_counts": {
            key: len(value)
            for key, value in manifest["audit"].items()
            if key.endswith("_groups")
        },
        "duplicate_group_sizes": {
            key: [len(group) for group in value]
            for key, value in manifest["audit"].items()
            if key.endswith("_groups")
        },
        "duplicate_masks_are_exactly_empty_masks": (
            len(manifest["audit"]["exact_duplicate_mask_groups"]) == 1
            and set(manifest["audit"]["exact_duplicate_mask_groups"][0])
            == {
                sample["id"]
                for sample in manifest["samples"]
                if sample["building_pixels"] == 0
            }
        ),
        "role_summaries": role_summaries,
        "split_counts": splits["counts"],
        "spatial_audit": splits["spatial_audit"],
        "future_pilot_subset": splits["future_pilot_subset"],
    }


def select_development_examples(manifest: dict) -> list[dict]:
    """Select four deterministic train/validation examples without test access."""
    development = [sample for sample in manifest["samples"] if sample["role"] in {"train", "val"}]
    fractions = {
        sample["id"]: sample["building_pixels"] / (512 * 512)
        for sample in development
    }
    training = [
        sample
        for sample in development
        if sample["role"] == "train" and sample["building_pixels"] > 0
    ]
    validation = [
        sample
        for sample in development
        if sample["role"] == "val" and sample["building_pixels"] > 0
    ]
    median_fraction = float(np.median([fractions[sample["id"]] for sample in training]))
    sparse_fraction = float(
        np.quantile([fractions[sample["id"]] for sample in validation], 0.1)
    )
    choices = [
        min(training, key=lambda sample: abs(fractions[sample["id"]] - median_fraction)),
        min(validation, key=lambda sample: abs(fractions[sample["id"]] - sparse_fraction)),
        max(validation, key=lambda sample: fractions[sample["id"]]),
    ]
    empty = next((sample for sample in development if sample["building_pixels"] == 0), None)
    if empty is not None:
        choices.append(empty)
    unique = {sample["id"]: sample for sample in choices}
    return [unique[sample_id] for sample_id in sorted(unique)]


def plot_whu_examples(manifest: dict, project_root: Path):
    """Plot aligned images, masks and overlays for selected development samples."""
    examples = select_development_examples(manifest)
    figure, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)), squeeze=False)
    for row, sample in enumerate(examples):
        image = read_whu_image(project_root / sample["image"])
        mask = load_whu_binary_target(read_whu_mask(project_root / sample["mask"]))
        overlay = image.astype(float) / 255
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([1.0, 0.2, 0.1])
        panels = (image, mask, overlay)
        titles = (
            f"{sample['id']} — RGB",
            f"Reference footprint ({mask.mean():.1%})",
            "Building overlay (red)",
        )
        for axis, panel, title in zip(axes[row], panels, titles):
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=1 if panel.ndim == 2 else None)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.suptitle("WHU development examples selected by mask-area criteria", fontsize=12)
    figure.tight_layout()
    return figure


def plot_provider_split_schematic():
    """Plot documented tile-block sizes without implying recovered coordinates."""
    colors = {"train": "#377eb8", "val": "#ff9f1c", "test": "#e41a1c"}
    figure, axes = plt.subplots(1, 4, figsize=(11, 3), constrained_layout=True)
    blocks = [
        ("Validation", "val", 14, 74),
        ("Training", "train", 64, 74),
        ("Test part 1", "test", 17, 42),
        ("Test part 2", "test", 23, 74),
    ]
    for axis, (title, role, width, height) in zip(axes, blocks):
        axis.add_patch(plt.Rectangle((0, 0), width, height, color=colors[role], alpha=0.82))
        axis.set_xlim(0, width)
        axis.set_ylim(0, 74)
        axis.set_aspect("equal")
        axis.set_title(f"{title}\n{width} × {height} tiles", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Provider-documented tile-block dimensions (schematic, positions not reconstructed)")
    return figure
