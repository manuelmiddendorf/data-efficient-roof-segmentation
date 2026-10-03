"""Report figures for the RID data audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .data import load_binary_target, read_image, read_mask


ROLE_COLOURS = {"training": "#0072B2", "validation": "#E69F00", "test": "#CC79A7", "excluded": "#999999"}


def representative_ids(manifest: dict, splits: dict, per_role: int = 3) -> list[tuple[str, str]]:
    """Choose label-area quantiles from development roles for a visual audit."""
    records = {sample["id"]: sample for sample in manifest["samples"]}
    chosen: list[tuple[str, str]] = []
    for role in ("training", "validation"):
        ids = splits["roles"][role]
        ordered = sorted(ids, key=lambda sample_id: records[sample_id]["roof_pixels"])
        indices = np.linspace(0, len(ordered) - 1, per_role, dtype=int)
        chosen.extend((role, ordered[index]) for index in indices)
    return chosen


def plot_examples(project_root: Path, manifest: dict, selections: list[tuple[str, str]]):
    """Plot RGB image, binary reference and overlay for selected development samples."""
    records = {sample["id"]: sample for sample in manifest["samples"]}
    figure, axes = plt.subplots(len(selections), 3, figsize=(9, 3 * len(selections)), squeeze=False)
    overlay_cmap = ListedColormap([(0, 0, 0, 0), (0.84, 0.15, 0.42, 0.55)])
    for row, (role, sample_id) in enumerate(selections):
        record = records[sample_id]
        image = read_image(project_root / record["image"])
        target = load_binary_target(read_mask(project_root / record["mask"]))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(target, cmap=overlay_cmap, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{role.title()} · ID {sample_id}\nroof {target.mean():.1%}")
    for axis, title in zip(axes[0], ("RGB image", "Binary roof reference", "Reference overlay")):
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Representative development images selected by roof-area quantiles", y=1.002)
    figure.tight_layout()
    return figure


def plot_split_map(manifest: dict, splits: dict):
    """Plot footprint centres by final role in geographic coordinates."""
    records = {sample["id"]: sample for sample in manifest["samples"]}
    figure, axis = plt.subplots(figsize=(8, 7))
    for role in ("excluded", "training", "validation", "test"):
        centres = []
        for sample_id in splits["roles"][role]:
            west, south, east, north = records[sample_id]["raster"]["bounds_wgs84"]
            centres.append(((west + east) / 2, (south + north) / 2))
        points = np.asarray(centres)
        axis.scatter(points[:, 0], points[:, 1], s=12, alpha=0.72, label=f"{role.title()} (n={len(points)})", color=ROLE_COLOURS[role])
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("RID image centres after strict cross-role footprint exclusions")
    axis.legend(frameon=False, markerscale=1.5)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    return figure
