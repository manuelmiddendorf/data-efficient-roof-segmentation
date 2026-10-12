"""Train exactly the paired Random and ImageNet WHU pilot models."""

from pathlib import Path

import torch

from roofseg.integrity import load_json
from roofseg.pilot import INITIALIZATIONS, build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.whu_data import build_whu_splits
from roofseg.whu_pilot import build_whu_pilot_config, whu_run_directory


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU provider roles and subset do not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU pilot requires available MPS on this host; selected {device}.")
print(f"Training device: {device}", flush=True)

test_ids = set(splits["roles"]["test"])
for initialization in INITIALIZATIONS:
    config = build_whu_pilot_config(splits, initialization)
    if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
        raise SystemExit("WHU training or validation inputs intersect provider test data.")
    metadata = build_metadata(ROOT, manifest, splits, config, device)
    directory = whu_run_directory(ROOT, splits, initialization)
    if prepare_run(directory, config, metadata):
        verify_completed_run(directory)
        print(f"Reusing compatible completed run: {directory.name}", flush=True)
        continue
    fit_run(ROOT, config, metadata, manifest, device, directory)
    verify_completed_run(directory)
