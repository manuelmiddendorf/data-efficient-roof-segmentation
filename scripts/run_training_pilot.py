"""Verify inputs once and execute the six fixed Stage 2 pilot runs."""

from pathlib import Path
import torch

from roofseg.audit import build_data_manifest, build_splits
from roofseg.integrity import load_json
from roofseg.pilot import (INITIALIZATIONS, PILOT_SIZES, build_config, build_metadata,
                           pilot_run_root, run_name)
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")

print("Verifying all provider-bound RID inputs once before the experiment block...", flush=True)
if build_data_manifest(ROOT) != manifest or build_splits(manifest, ROOT) != splits:
    raise SystemExit("Raw RID data or saved split differs from the Stage 1 references.")

device = select_device()
print(f"Training device: {device}", flush=True)
pilot_ids = set(splits["roles"]["validation"])
for size in PILOT_SIZES:
    pilot_ids.update(splits["training_subsets"]["repetitions"]["1"]["subsets"][str(size)])
if pilot_ids & set(splits["roles"]["test"]):
    raise SystemExit("Pilot inputs intersect the locked test role.")

run_root = pilot_run_root(ROOT, splits)
for size in PILOT_SIZES:
    for initialization in INITIALIZATIONS:
        config = build_config(splits, size, initialization)
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        directory = run_root / run_name(size, initialization)
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
