"""Build the shared RID data manifest, geographic split and audit summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import RIDDataset, check_role_partition, parse_split_file, raster_metadata, read_mask
from .integrity import read_provider_checksums, sha256_file, verify_files
from .spatial import (metric_footprints, overlapping_pairs, remove_crossing_training_images,
                      select_nested_spatial_subsets)


RELEASE_PREFIXES = (
    "RID_dataset/images_roof_centered_geotiff/",
    "RID_dataset/masks_segments_reviewed/",
    "RID_dataset/filenames_train_val_test_split/",
)


def _release_paths(references: dict[str, str]) -> list[str]:
    selected = [path for path in references if path == "RID_dataset/README_data.md"
                or path.startswith(RELEASE_PREFIXES)]
    expected = {"images": 1880, "masks": 1880, "splits": 15}
    actual = {
        "images": sum(path.startswith(RELEASE_PREFIXES[0]) for path in selected),
        "masks": sum(path.startswith(RELEASE_PREFIXES[1]) for path in selected),
        "splits": sum(path.startswith(RELEASE_PREFIXES[2]) for path in selected),
    }
    if actual != expected or "RID_dataset/README_data.md" not in selected:
        raise ValueError(f"Provider checksum inventory differs from RID 1.0: {actual}")
    return selected


def build_data_manifest(project_root: Path) -> dict:
    """Verify RID 1.0 and describe every paired image and reviewed segment mask."""
    release_root = project_root / "data/raw/rid"
    checksum_path = release_root / "Checksums.sha256"
    references = read_provider_checksums(checksum_path)
    selected_paths = _release_paths(references)
    verify_files(release_root, references, selected_paths)
    dataset = RIDDataset(release_root)
    pairs = dataset.discover()
    if len(pairs) != 1880:
        raise ValueError(f"Expected 1880 RID pairs, found {len(pairs)}.")

    def inspect(pair):
        image_relative = pair.image_path.relative_to(release_root).as_posix()
        mask_relative = pair.mask_path.relative_to(release_root).as_posix()
        mask = read_mask(pair.mask_path)
        counts = np.bincount(mask.ravel(), minlength=18)
        return {
            "id": pair.sample_id,
            "image": f"data/raw/rid/{image_relative}",
            "mask": f"data/raw/rid/{mask_relative}",
            "image_sha256": references[image_relative],
            "mask_sha256": references[mask_relative],
            "raster": raster_metadata(pair.image_path),
            "mask_mode": "L",
            "mask_dtype": "uint8",
            "mask_values": np.flatnonzero(counts).astype(int).tolist(),
            "background_pixels": int(counts[17]),
            "roof_pixels": int(counts[:17].sum()),
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        samples = list(executor.map(inspect, pairs))
    samples.sort(key=lambda sample: int(sample["id"]))

    split_root = release_root / "RID_dataset/filenames_train_val_test_split"
    provider_splits: dict[str, dict[str, list[str]]] = {}
    all_ids = [sample["id"] for sample in samples]
    for split_number in range(1, 6):
        roles = {role: parse_split_file(split_root / f"{role}_filenames_{split_number}_rev.txt")
                 for role in ("train", "val", "test")}
        check_role_partition(roles, all_ids)
        provider_splits[str(split_number)] = roles

    image_hash_groups: dict[str, list[str]] = defaultdict(list)
    mask_hash_groups: dict[str, list[str]] = defaultdict(list)
    mask_pixels = Counter()
    for sample in samples:
        image_hash_groups[sample["image_sha256"]].append(sample["id"])
        mask_hash_groups[sample["mask_sha256"]].append(sample["id"])
        mask_pixels["background"] += sample["background_pixels"]
        mask_pixels["roof"] += sample["roof_pixels"]
    duplicate_images = [sorted(ids, key=int) for ids in image_hash_groups.values() if len(ids) > 1]
    duplicate_masks = [sorted(ids, key=int) for ids in mask_hash_groups.values() if len(ids) > 1]
    crs_values = sorted({sample["raster"]["crs"] for sample in samples})
    shapes = sorted({(sample["raster"]["height"], sample["raster"]["width"],
                      sample["raster"]["bands"]) for sample in samples})
    mask_values = sorted({value for sample in samples for value in sample["mask_values"]})
    return {
        "schema_version": 1,
        "dataset": {
            "name": "RID – Roof Information Dataset",
            "release": "1.0",
            "issued": "2022-05-11",
            "doi": "10.14459/2022mp1655470",
            "record_url": "https://mediatum.ub.tum.de/1655470",
            "provider_checksum_file": "data/raw/rid/Checksums.sha256",
            "provider_checksum_file_sha256": sha256_file(checksum_path),
            "download_transport": "rsync://m1655470@dataserv.ub.tum.de/m1655470/",
            "upstream_code_commit": "6f5d084b699ced0963a0a414b2746d85de50bdb4",
            "dataset_terms": "CC BY-NC (as stated in RID_dataset/README_data.md)",
            "imagery_notice": "Google Maps Static API imagery; provider limits use to non-commercial research/education and related fair-use purposes.",
        },
        "label_contract": {
            "product": "masks_segments_reviewed",
            "background_codes": [17],
            "roof_codes": list(range(17)),
            "roof_code_meaning": "codes 0--15 are azimuth classes; code 16 is flat roof",
            "binary_mapping": "roof = mask != 17",
            "mapping_evidence": "RID mask_generation.py initializes background to len(label_classes)=17; definitions.py maps codes 0--16 to 16 azimuth classes plus flat roof.",
            "unknown_codes": [],
            "validity": "All 512x512 mask pixels have a provider-defined class code; imagery contains no alpha band or nodata declaration.",
            "caveat": "An explicit class code does not prove that every visible roof pixel is annotated correctly or completely.",
        },
        "audit": {
            "pair_count": len(samples),
            "image_shapes_hwc": [list(shape) for shape in shapes],
            "image_crs": crs_values,
            "mask_values_observed": mask_values,
            "background_pixels": mask_pixels["background"],
            "roof_pixels": mask_pixels["roof"],
            "roof_fraction": mask_pixels["roof"] / (mask_pixels["background"] + mask_pixels["roof"]),
            "all_image_pixels_valid": all(sample["raster"]["valid_pixels"] == 512 * 512 for sample in samples),
            "exact_duplicate_image_groups": duplicate_images,
            "exact_duplicate_mask_groups": duplicate_masks,
            "provider_split_counts": {
                number: {role: len(ids) for role, ids in roles.items()}
                for number, roles in provider_splits.items()
            },
            "provider_test_ids_identical_across_splits": len({tuple(sorted(roles["test"], key=int)) for roles in provider_splits.values()}) == 1,
        },
        "samples": samples,
    }


def build_splits(manifest: dict, project_root: Path) -> dict:
    """Adopt provider split D1 and exclude every positive-area cross-role footprint."""
    samples = manifest["samples"]
    footprints = metric_footprints(samples)
    release_root = project_root / "data/raw/rid/RID_dataset/filenames_train_val_test_split"
    all_ids = [sample["id"] for sample in samples]
    official = {role: parse_split_file(release_root / f"{role}_filenames_1_rev.txt")
                for role in ("train", "val", "test")}
    check_role_partition(official, all_ids)

    validation_test_pairs = overlapping_pairs(footprints, official["val"], official["test"])
    validation_crossing = {pair["left"] for pair in validation_test_pairs}
    validation = sorted(set(official["val"]) - validation_crossing, key=int)
    training, training_exclusions = remove_crossing_training_images(
        footprints, official["train"], {"test": official["test"], "validation": validation}
    )
    duplicate_exclusions = []
    for group in manifest["audit"]["exact_duplicate_image_groups"]:
        retained_group = [sample_id for sample_id in group if sample_id in set(training) | set(validation)]
        if len(retained_group) < 2:
            continue
        canonical = min(retained_group, key=int)
        for sample_id in retained_group:
            if sample_id == canonical:
                continue
            training = [value for value in training if value != sample_id]
            validation = [value for value in validation if value != sample_id]
            duplicate_exclusions.append({"id": sample_id, "reason": f"exact_duplicate_of_{canonical}"})
    exclusions = ([{"id": sample_id, "reason": "validation_footprint_intersects_test"}
                   for sample_id in sorted(validation_crossing, key=int)]
                  + training_exclusions + duplicate_exclusions)
    exclusions.sort(key=lambda row: int(row["id"]))
    excluded_ids = [row["id"] for row in exclusions]
    roles = {"training": training, "validation": validation,
             "test": sorted(official["test"], key=int), "excluded": excluded_ids}
    check_role_partition(roles, all_ids)
    after = {
        "training_validation": overlapping_pairs(footprints, training, validation),
        "training_test": overlapping_pairs(footprints, training, official["test"]),
        "validation_test": overlapping_pairs(footprints, validation, official["test"]),
    }
    if any(after.values()):
        raise AssertionError("Positive-area geographic overlap remains after exclusions.")
    subset_sizes = [size for size in (25, 50, 100, 250, 500) if size < len(training)]
    repetitions = {}
    for repetition, seed in enumerate((17, 29, 43), 1):
        subsets = select_nested_spatial_subsets(footprints, training, subset_sizes, seed)
        subsets["full"] = training
        repetitions[str(repetition)] = {"seed": seed, "subsets": subsets}
    return {
        "schema_version": 1,
        "dataset_manifest": "data/metadata/data_manifest.json",
        "split_name": "rid_d1_strict_footprints_v1",
        "source": {
            "provider_split": "D1 / north (files ending _1_rev.txt)",
            "rationale": "Use the provider's northern validation block and shared buffered test IDs, remove every positive-area cross-role footprint overlap, and retain only one copy from each exact duplicate group within development roles.",
            "metric_crs": "EPSG:25832",
            "intersection_tolerance_m2": 0.01,
            "test_policy": "All 154 provider test IDs are locked outside development and subset selection.",
        },
        "roles": roles,
        "exclusions": exclusions,
        "counts": {role: len(ids) for role, ids in roles.items()},
        "spatial_audit": {
            "before": {
                "training_validation_pair_count": len(overlapping_pairs(footprints, official["train"], official["val"])),
                "training_test_pair_count": len(overlapping_pairs(footprints, official["train"], official["test"])),
                "validation_test_pair_count": len(validation_test_pairs),
            },
            "after": {name + "_pair_count": len(pairs) for name, pairs in after.items()},
            "method": "Pairwise intersections of complete georeferenced 512x512 image footprints; positive area above 0.01 m² counts as overlap.",
            "repeated_building_limit": "Disjoint footprints prevent the same visible building from appearing across roles, but do not establish independence of nearby architecture, capture conditions or map-source processing.",
        },
        "training_subsets": {
            "method": "Seeded maximin sampling of image-footprint centres within the retained training pool; subsets are nested within each repetition.",
            "uses_labels_or_held_out_data": False,
            "repetitions": repetitions,
        },
    }


def build_report(manifest: dict, splits: dict) -> dict:
    """Return compact, notebook-facing audit values from the two shared records."""
    samples = manifest["samples"]
    fractions = np.asarray([sample["roof_pixels"] / (512 * 512) for sample in samples])
    return {
        "dataset": manifest["dataset"],
        "label_contract": manifest["label_contract"],
        "audit": manifest["audit"],
        "split_counts": splits["counts"],
        "spatial_audit": splits["spatial_audit"],
        "roof_fraction_summary": {
            "minimum": float(fractions.min()),
            "median": float(np.median(fractions)),
            "mean": float(fractions.mean()),
            "maximum": float(fractions.max()),
            "empty_masks": int(np.count_nonzero(fractions == 0)),
            "full_masks": int(np.count_nonzero(fractions == 1)),
        },
    }
