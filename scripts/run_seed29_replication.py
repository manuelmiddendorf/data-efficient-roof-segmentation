"""Run the two fresh Seed 29 fixed-drop replication configurations."""

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


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Stage 3 training requires available MPS on this host; selected {device}.")
print(f"Training device: {device}", flush=True)

for initialization in INITIALIZATIONS:
    seed17_config = build_late_lr_drop_config(splits, initialization, 1)
    seed17_directory = late_lr_drop_run_directory(ROOT, splits, initialization, 1)
    if json.loads((seed17_directory / "config.json").read_text()) != seed17_config:
        raise SystemExit(f"Seed 17 reference differs: {seed17_directory.name}")
    verify_completed_run(seed17_directory)

    config = build_late_lr_drop_config(splits, initialization, 2)
    if set(config["training_ids"]) & (
        set(splits["roles"]["validation"]) | set(splits["roles"]["test"])
    ):
        raise SystemExit("Seed 29 training inputs intersect validation or test roles.")
    metadata = build_metadata(ROOT, manifest, splits, config, device)
    directory = late_lr_drop_run_directory(ROOT, splits, initialization, 2)
    if prepare_run(directory, config, metadata):
        verify_completed_run(directory)
        print(f"Reusing compatible completed run: {directory.name}", flush=True)
        continue
    fit_run(ROOT, config, metadata, manifest, device, directory)
    verify_completed_run(directory)
