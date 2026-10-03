"""Geographic footprint checks and deterministic training-subset selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree


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
