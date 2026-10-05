"""Run the four fresh n=25 strong-weight-decay configurations."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.weight_decay import (
    SUBSET_REPETITIONS,
    build_weight_decay_config,
    weight_decay_run_directory,
)

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Weight-decay training requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)

for repetition in SUBSET_REPETITIONS:
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
        reference_directory = late_lr_drop_run_directory(
            ROOT, splits, initialization, repetition, 25
        )
        if json.loads((reference_directory / "config.json").read_text()) != reference:
            raise SystemExit(f"Reference differs: {reference_directory.name}")
        verify_completed_run(reference_directory)

        config = build_weight_decay_config(splits, initialization, repetition)
        if set(config["training_ids"]) & (
            set(splits["roles"]["validation"]) | set(splits["roles"]["test"])
        ):
            raise SystemExit("Training inputs intersect validation or test roles.")
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        directory = weight_decay_run_directory(ROOT, splits, initialization, repetition)
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
