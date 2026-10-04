"""Metadata-only construction and diagnostics for a candidate geographic split."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
import numpy as np

from .plots import ROLE_COLOURS
from .spatial import (APPROVED_TEST_BOUNDS_M, assign_compact_geographic_roles,
                      directional_nearest_distances, metric_footprints)


PROPOSAL_CRS = "EPSG:25832"


def prior_development_usage(project_root: Path, manifest: dict, splits: dict) -> dict:
    """Reconstruct model and qualitative use from saved, executed artifacts.

    File-integrity, coordinate, label-code and aggregate label audits are listed
    separately because they did not expose imagery to model selection or error
    analysis.
    """
    run_records = []
    model_training_ids: set[str] = set()
    model_validation_ids: set[str] = set()
    run_directories = sorted((project_root / "runs/training_pilot").glob("*"))
    run_directories += sorted((project_root / "runs/training_pilot_interrupted").glob("*"))
    for directory in run_directories:
        config_path = directory / "config.json"
        history_path = directory / "history.csv"
        summary_path = directory / "summary.json"
        if not (config_path.exists() and history_path.exists() and summary_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        summary = json.loads(summary_path.read_text())
        history_rows = max(0, len(history_path.read_text().splitlines()) - 1)
        if history_rows:
            model_training_ids.update(config["training_ids"])
            model_validation_ids.update(config["validation_ids"])
        run_records.append({
            "run": directory.relative_to(project_root).as_posix(),
            "status": summary.get("status"),
            "history_rows": history_rows,
            "configured_training_images": len(config["training_ids"]),
            "configured_validation_images": len(config["validation_ids"]),
        })

    preflight_path = project_root / "reports/training_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight_learning_ids = set(preflight["tiny_learning"]["ids"])
    preflight_learning_ids.add(preflight["real_data_contract"]["sample_id"])

    records = {sample["id"]: sample for sample in manifest["samples"]}
    qualitative_ids: set[str] = set()
    qualitative_rows = []
    for role in ("training", "validation"):
        ordered = sorted(splits["roles"][role], key=lambda sample_id: records[sample_id]["roof_pixels"])
        for index in np.linspace(0, len(ordered) - 1, 3, dtype=int):
            sample_id = ordered[index]
            qualitative_ids.add(sample_id)
            qualitative_rows.append({"id": sample_id, "old_role": role})

    return {
        "executed_runs": run_records,
        "model_training_ids": sorted(model_training_ids, key=int),
        "model_validation_ids": sorted(model_validation_ids, key=int),
        "preflight_learning_ids": sorted(preflight_learning_ids, key=int),
        "qualitative_development_examples": sorted(qualitative_rows, key=lambda row: int(row["id"])),
        "model_or_qualitative_ids": sorted(
            model_training_ids | model_validation_ids | preflight_learning_ids | qualitative_ids,
            key=int,
        ),
        "technical_audit": {
            "sample_count": len(manifest["samples"]),
            "uses": [
                "provider checksum and pairing verification",
                "raster coordinates and complete-footprint calculations",
                "mask-code validation and aggregate roof-area summaries",
            ],
            "interpretation": "Technical audit only; not model fitting, checkpoint selection or qualitative error analysis.",
        },
    }


def build_split_proposal(manifest: dict, old_splits: dict, usage: dict) -> dict:
    """Build one southwest test-area proposal without reading imagery or scores."""
    footprints = metric_footprints(manifest["samples"], PROPOSAL_CRS)
    old_validation = set(old_splits["roles"]["validation"])
    exposed = set(usage["model_or_qualitative_ids"])
    roles, exclusions, overlap_counts = assign_compact_geographic_roles(
        footprints, old_validation, manifest["audit"]["exact_duplicate_image_groups"]
    )
    test = set(roles["test"])
    old_role = {sample_id: role for role, ids in old_splits["roles"].items() for sample_id in ids}
    transitions = Counter((old_role[sample_id], role) for role, ids in roles.items() for sample_id in ids)
    old_test = set(old_splits["roles"]["test"])
    test_bounds = APPROVED_TEST_BOUNDS_M
    return {
        "status": "proposal_only_not_adopted",
        "do_not_use_for_training": True,
        "name": "southwest_compact_test_candidate",
        "selection_basis": "Complete metric footprints within a southwest coordinate window; no imagery, scores, prediction errors or label distribution used.",
        "crs": PROPOSAL_CRS,
        "test_area_bounds_m": list(test_bounds),
        "test_area_width_m": test_bounds[2] - test_bounds[0],
        "test_area_height_m": test_bounds[3] - test_bounds[1],
        "roles": roles,
        "counts": {role: len(ids) for role, ids in roles.items()},
        "exclusions": exclusions,
        "exclusion_reason_counts": dict(sorted(Counter(
            reason for row in exclusions for reason in row["reasons"]
        ).items())),
        "cross_role_positive_area_overlaps": overlap_counts,
        "prior_use_intersection": {
            "proposed_test_images_previously_used": sorted(test & exposed, key=int),
            "model_training": sorted(test & set(usage["model_training_ids"]), key=int),
            "model_validation": sorted(test & set(usage["model_validation_ids"]), key=int),
            "qualitative_examples": sorted(
                test & {row["id"] for row in usage["qualitative_development_examples"]}, key=int
            ),
        },
        "old_provider_test_transition": {
            role: len(old_test & set(ids)) for role, ids in roles.items()
        },
        "old_to_proposed_role_counts": [
            {"old_role": old, "proposed_role": new, "images": count}
            for (old, new), count in sorted(transitions.items())
        ],
        "distance_summary_m": cross_role_distances(footprints, roles),
    }


def cross_role_distances(footprints: dict[str, object], roles: dict[str, list[str]]) -> list[dict]:
    """Summarize nearest complete-footprint distances between proposed roles."""
    rows = []
    for left_role, right_role in (("training", "validation"), ("training", "test"), ("validation", "test")):
        rows.append({
            "roles": f"{left_role}–{right_role}",
            **directional_nearest_distances(footprints, roles[left_role], roles[right_role]),
        })
    return rows


def plot_split_proposal(manifest: dict, proposal: dict):
    """Plot proposal-wide roles and a full-footprint zoom at the test boundary."""
    footprints = metric_footprints(manifest["samples"], proposal["crs"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    colours = {**ROLE_COLOURS, "test": "#009E73"}

    for role in ("excluded", "training", "validation", "test"):
        ids = proposal["roles"][role]
        centres = np.asarray([[footprints[sample_id].centroid.x, footprints[sample_id].centroid.y]
                              for sample_id in ids])
        axes[0].scatter(centres[:, 0], centres[:, 1], s=12, alpha=0.72,
                        color=colours[role], label=f"{role.title()} (n={len(ids)})")
    west, south, east, north = proposal["test_area_bounds_m"]
    axes[0].add_patch(Rectangle((west, south), east - west, north - south,
                                fill=False, color="#006D5B", linewidth=2.2,
                                label="Southwest test selection window"))
    axes[0].set_title("Proposed roles across Wartenberg")
    axes[0].legend(frameon=False, fontsize=8)

    zoom_ids = [sample_id for sample_id, geometry in footprints.items()
                if geometry.bounds[0] < east + 130 and geometry.bounds[1] < north + 130]
    role_by_id = {sample_id: role for role, ids in proposal["roles"].items() for sample_id in ids}
    for sample_id in zoom_ids:
        coordinates = np.asarray(footprints[sample_id].exterior.coords)
        role = role_by_id[sample_id]
        axes[1].add_patch(PolygonPatch(coordinates, closed=True, facecolor=colours[role],
                                      edgecolor="white", linewidth=0.25, alpha=0.62))
    axes[1].add_patch(Rectangle((west, south), east - west, north - south,
                                fill=False, color="#006D5B", linewidth=2.2))
    axes[1].set_xlim(west - 20, east + 130)
    axes[1].set_ylim(south - 20, north + 130)
    axes[1].set_title("Complete footprints at the southwest boundary")

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Easting in EPSG:25832 (m)")
        axis.set_ylabel("Northing in EPSG:25832 (m)")
        axis.grid(alpha=0.15)
    figure.suptitle("Metadata-only proposal: compact southwest test area", fontsize=14)
    figure.tight_layout()
    return figure
