"""Geographic footprint checks and deterministic training-subset selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree


APPROVED_TEST_BOUNDS_M = (720_200.0, 5_364_860.0, 720_800.0, 5_365_350.0)


def metric_footprints(samples: Iterable[dict], target_crs: str = "EPSG:25832") -> dict[str, object]:
    """Create metric image-footprint polygons from manifest WGS84 bounds."""
    project = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
    return {sample["id"]: transform(project, box(*sample["raster"]["bounds_wgs84"])) for sample in samples}


def overlapping_pairs(
    footprints: dict[str, object],
    left_ids: Iterable[str],
    right_ids: Iterable[str],
    minimum_area_m2: float = 0.01,
) -> list[dict]:
    """Find positive-area intersections across two disjoint image-ID sets."""
    left = sorted(set(left_ids), key=int)
    right = sorted(set(right_ids), key=int)
    if set(left) & set(right):
        raise ValueError("Spatial comparison roles must have disjoint IDs.")
    right_geometries = [footprints[sample_id] for sample_id in right]
    tree = STRtree(right_geometries)
    pairs: list[dict] = []
    for left_id in left:
        geometry = footprints[left_id]
        for index in tree.query(geometry, predicate="intersects"):
            right_id = right[int(index)]
            area = geometry.intersection(right_geometries[int(index)]).area
            if area > minimum_area_m2:
                pairs.append({"left": left_id, "right": right_id, "area_m2": float(area)})
    return sorted(pairs, key=lambda row: (int(row["left"]), int(row["right"])))


def remove_crossing_training_images(
    footprints: dict[str, object], train_ids: Iterable[str], protected_roles: dict[str, Iterable[str]]
) -> tuple[list[str], list[dict]]:
    """Exclude training footprints intersecting validation or test footprints."""
    training = set(train_ids)
    exclusions: dict[str, set[str]] = defaultdict(set)
    for role, ids in protected_roles.items():
        pairs = overlapping_pairs(footprints, training, ids)
        for pair in pairs:
            exclusions[pair["left"]].add(role)
    remaining = training - set(exclusions)
    rows = [{"id": sample_id,
             "reason": "footprint_intersects_" + "_and_".join(sorted(roles))}
            for sample_id, roles in sorted(exclusions.items(), key=lambda item: int(item[0]))]
    return sorted(remaining, key=int), rows


def select_nested_spatial_subsets(
    footprints: dict[str, object], pool_ids: Iterable[str], sizes: Iterable[int], seed: int
) -> dict[str, list[str]]:
    """Select nested, spatially distributed subsets by seeded maximin centres.

    The first sample is seeded; each later sample maximizes its distance to the
    nearest selected centre. Ties use numeric sample ID. Only training-pool
    coordinates are used, never masks, validation values or test appearance.
    """
    ids = sorted(set(pool_ids), key=int)
    requested = sorted(set(int(size) for size in sizes))
    if not ids or not requested or requested[-1] > len(ids) or requested[0] < 1:
        raise ValueError("Subset sizes must lie within the nonempty training pool.")
    centres = np.asarray([[footprints[sample_id].centroid.x, footprints[sample_id].centroid.y]
                          for sample_id in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(ids)))
    selected = [first]
    nearest = np.sum((centres - centres[first]) ** 2, axis=1)
    nearest[first] = -1
    while len(selected) < requested[-1]:
        maximum = nearest.max()
        candidates = np.flatnonzero(np.isclose(nearest, maximum, rtol=0, atol=1e-9))
        next_index = min(candidates, key=lambda index: int(ids[int(index)]))
        selected.append(int(next_index))
        nearest = np.minimum(nearest, np.sum((centres - centres[next_index]) ** 2, axis=1))
        nearest[selected] = -1
    ordered = [ids[index] for index in selected]
    return {str(size): sorted(ordered[:size], key=int) for size in requested}


def assign_compact_geographic_roles(
    footprints: dict[str, object],
    validation_ids: Iterable[str],
    duplicate_groups: Iterable[Iterable[str]],
    test_bounds_m: tuple[float, float, float, float] = APPROVED_TEST_BOUNDS_M,
) -> tuple[dict[str, list[str]], list[dict], dict[str, int]]:
    """Assign the approved compact test region and remove cross-role overlaps.

    Complete footprints contained by ``test_bounds_m`` enter the test role.
    Footprints crossing that coordinate boundary are excluded. The supplied
    northern validation role is retained, one canonical copy per exact-duplicate
    group is kept, and remaining training footprints that overlap protected
    roles by positive area are excluded. No distance buffer is applied.
    """
    all_ids = set(footprints)
    validation = set(validation_ids)
    selection_area = box(*test_bounds_m)
    reasons: dict[str, set[str]] = defaultdict(set)
    retained = set(all_ids)

    for group_values in duplicate_groups:
        group = list(group_values)
        keep = min(group, key=lambda sample_id: (0 if sample_id in validation else 1, int(sample_id)))
        for sample_id in group:
            if sample_id != keep:
                retained.discard(sample_id)
                reasons[sample_id].add(f"exact_duplicate_of_{keep}")

    inside = {sample_id for sample_id, geometry in footprints.items() if selection_area.covers(geometry)}
    boundary = {
        sample_id for sample_id, geometry in footprints.items()
        if geometry.intersects(selection_area) and sample_id not in inside
    }
    for sample_id in boundary:
        retained.discard(sample_id)
        reasons[sample_id].add("southwest_test_boundary_straddler")

    test = (inside & retained) - validation
    validation &= retained
    if overlapping_pairs(footprints, test, validation):
        raise AssertionError("Approved southwest test and northern validation footprints overlap.")

    training, crossing_rows = remove_crossing_training_images(
        footprints,
        retained - test - validation,
        {"test": test, "validation": validation},
    )
    for row in crossing_rows:
        reasons[row["id"]].add(row["reason"])

    training_set = set(training)
    excluded = all_ids - training_set - validation - test
    roles = {
        "training": sorted(training_set, key=int),
        "validation": sorted(validation, key=int),
        "test": sorted(test, key=int),
        "excluded": sorted(excluded, key=int),
    }
    assigned = [sample_id for ids in roles.values() for sample_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise AssertionError("Geographic roles do not form a complete disjoint partition.")

    overlap_counts = {
        "training_validation": len(overlapping_pairs(footprints, training_set, validation)),
        "training_test": len(overlapping_pairs(footprints, training_set, test)),
        "validation_test": len(overlapping_pairs(footprints, validation, test)),
    }
    if any(overlap_counts.values()):
        raise AssertionError("Positive-area cross-role footprint overlap remains.")
    exclusions = [
        {"id": sample_id, "reasons": sorted(reasons[sample_id])}
        for sample_id in sorted(excluded, key=int)
    ]
    return roles, exclusions, overlap_counts


def directional_nearest_distances(
    footprints: dict[str, object], from_ids: Iterable[str], to_ids: Iterable[str]
) -> dict[str, float]:
    """Summarize distance from each source footprint to the nearest target footprint."""
    source = list(from_ids)
    target_geometries = [footprints[sample_id] for sample_id in to_ids]
    if not source or not target_geometries:
        raise ValueError("Directional distance roles must both be nonempty.")
    tree = STRtree(target_geometries)
    distances = []
    for sample_id in source:
        geometry = footprints[sample_id]
        nearest_index = int(tree.nearest(geometry))
        distances.append(geometry.distance(target_geometries[nearest_index]))
    values = np.asarray(distances)
    return {
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
    }
