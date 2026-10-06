"""Run the eight fixed n=25 boundary-weighted BCE configurations."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.boundary_weighted import (
    SUBSET_REPETITIONS,
    VARIANTS,
    boundary_run_directory,
    build_boundary_config,
    training_band_fractions,
)
from roofseg.integrity import load_json
from roofseg.optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.training_data import RIDTensorStore

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Boundary-weighted training requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)
store = RIDTensorStore(ROOT, manifest)

for repetition in SUBSET_REPETITIONS:
    training_ids = splits["training_subsets"]["repetitions"][str(repetition)]["subsets"]["25"]
    fractions = training_band_fractions(store, training_ids)
    mean_fraction = sum(fractions.values()) / len(fractions)
    for variant in VARIANTS:
        for initialization in INITIALIZATIONS:
            reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
            reference_directory = late_lr_drop_run_directory(ROOT, splits, initialization, repetition, 25)
            if json.loads((reference_directory / "config.json").read_text()) != reference:
                raise SystemExit(f"Reference differs: {reference_directory.name}")
            verify_completed_run(reference_directory)
            config = build_boundary_config(
                splits, initialization, repetition, variant, mean_fraction
            )
            if set(config["training_ids"]) & (
                set(splits["roles"]["validation"]) | set(splits["roles"]["test"])
            ):
                raise SystemExit("Training inputs intersect validation or test roles.")
            metadata = build_metadata(ROOT, manifest, splits, config, device)
            directory = boundary_run_directory(
                ROOT, splits, initialization, repetition, variant
            )
            if prepare_run(directory, config, metadata):
                verify_completed_run(directory)
                print(f"Reusing compatible completed run: {directory.name}", flush=True)
                continue
            fit_run(ROOT, config, metadata, manifest, device, directory)
            verify_completed_run(directory)
