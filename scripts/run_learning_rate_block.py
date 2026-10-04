"""Verify Stage 2 references and run the four new Stage 3 learning-rate candidates."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import (
    INITIALIZATIONS,
    NEW_LEARNING_RATES,
    REFERENCE_LEARNING_RATE,
    build_learning_rate_config,
    learning_rate_run_directory,
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

test_ids = set(splits["roles"]["test"])
for initialization in INITIALIZATIONS:
    reference_config = build_learning_rate_config(
        splits, initialization, REFERENCE_LEARNING_RATE
    )
    if set(reference_config["training_ids"]) & test_ids:
        raise SystemExit("Training inputs intersect the locked test role.")
    if set(reference_config["validation_ids"]) & test_ids:
        raise SystemExit("Validation inputs intersect the locked test role.")
    reference_directory = learning_rate_run_directory(
        ROOT, splits, initialization, REFERENCE_LEARNING_RATE
    )
    saved_config = json.loads((reference_directory / "config.json").read_text())
    saved_metadata = json.loads((reference_directory / "metadata.json").read_text())
    expected_metadata = build_metadata(
        ROOT, manifest, splits, reference_config, device
    )
    if saved_config != reference_config:
        raise SystemExit(f"Reference configuration differs: {reference_directory.name}")
    if saved_metadata["compatibility_identity"] != expected_metadata["compatibility_identity"]:
        raise SystemExit(f"Reference input identity differs: {reference_directory.name}")
    verify_completed_run(reference_directory)
    print(f"Verified Stage 2 reference: {reference_directory.name}", flush=True)

for initialization in INITIALIZATIONS:
    for learning_rate in NEW_LEARNING_RATES:
        config = build_learning_rate_config(splits, initialization, learning_rate)
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        directory = learning_rate_run_directory(
            ROOT, splits, initialization, learning_rate
        )
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
