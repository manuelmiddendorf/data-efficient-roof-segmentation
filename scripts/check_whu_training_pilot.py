"""Run the bounded real-data preflight for the paired WHU pilot."""

from pathlib import Path
import json
import tempfile

import torch

from roofseg.integrity import load_json
from roofseg.model import build_model, state_digest
from roofseg.pilot import build_metadata
from roofseg.training import (
    adamw, evaluate, restore_checkpoint, select_device, synchronize, train_step,
)
from roofseg.training_data import WHUTensorStore, paired_schedule
from roofseg.whu_data import build_whu_splits, read_whu_mask
from roofseg.whu_pilot import build_whu_pilot_config


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU provider roles and subset do not reproduce exactly.")

configs = {
    initialization: build_whu_pilot_config(splits, initialization)
    for initialization in ("random", "imagenet")
}
random_config, imagenet_config = configs["random"], configs["imagenet"]
if random_config["training_ids"] != imagenet_config["training_ids"]:
    raise AssertionError("Paired WHU training IDs differ.")
if set(random_config["training_ids"]) & set(splits["roles"]["val"]):
    raise AssertionError("WHU training subset intersects validation.")
if (set(random_config["training_ids"]) | set(random_config["validation_ids"])) & set(splits["roles"]["test"]):
    raise AssertionError("WHU pilot inputs intersect provider test data.")

schedule_random, schedule_record = paired_schedule(
    random_config["training_ids"], random_config["max_steps"], random_config["batch_size"],
    random_config["seeds"]["data_order"], random_config["seeds"]["augmentation"],
)
schedule_imagenet, _ = paired_schedule(
    imagenet_config["training_ids"], imagenet_config["max_steps"], imagenet_config["batch_size"],
    imagenet_config["seeds"]["data_order"], imagenet_config["seeds"]["augmentation"],
)
if schedule_random != schedule_imagenet:
    raise AssertionError("WHU data-order or D4 schedules differ between initializations.")

store = WHUTensorStore(ROOT, manifest)
empty_id = next(sample["id"] for sample in manifest["samples"] if sample["role"] == "train" and sample["building_pixels"] == 0)
positive_ids = [
    sample["id"] for sample in manifest["samples"]
    if sample["id"] in set(random_config["training_ids"]) and sample["building_pixels"] > 0
][:2]
_, empty_target = store.load_raw(empty_id)
if empty_target.any().item() or set(read_whu_mask(ROOT / store.records[empty_id]["mask"]).ravel()) != {1}:
    raise AssertionError("An all-background WHU mask did not map to an empty target.")
image, target = store.load_raw(positive_ids[0])
for code in range(8):
    batch = store.batch([positive_ids[0]], [code])
    if int(batch["target"].sum()) != int(target.sum()):
        raise AssertionError("D4 changed WHU target area.")

device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU pilot requires available MPS on this host; selected {device}.")
random_model, random_initialization = build_model("random", random_config["seeds"]["model"])
imagenet_model, imagenet_initialization = build_model("imagenet", imagenet_config["seeds"]["model"])
if random_initialization["decoder_sha256"] != imagenet_initialization["decoder_sha256"]:
    raise AssertionError("Paired decoder initializations differ.")
if random_initialization["encoder_sha256"] == imagenet_initialization["encoder_sha256"]:
    raise AssertionError("Random and ImageNet encoder states unexpectedly match.")
del random_model

model = imagenet_model.to(device)
optimizer, optimizer_record = adamw(model, 1e-3, 1e-4)
real_batch = store.batch(positive_ids, [0, 7])
values = train_step(model, optimizer, real_batch, device)
synchronize(device)
if not all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None):
    raise FloatingPointError("WHU MPS gradients are non-finite.")
with torch.inference_mode():
    output = model(real_batch["image"].to(device))
if output.shape != (2, 1, 512, 512) or output.device.type != "mps":
    raise AssertionError("WHU MPS forward pass violates the tensor contract.")

model, _ = build_model("imagenet", imagenet_config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, 1e-3, 1e-4)
tiny_batch = store.batch(positive_ids)
curve = []
for step in range(61):
    if step:
        train_step(model, optimizer, tiny_batch, device)
    if step % 10 == 0:
        _, metrics = evaluate(model, store, positive_ids, 2, device)
        curve.append({"step": step, "loss": metrics["loss"], "mean_iou": metrics["mean_iou"]})
        print(f"Tiny WHU learning step {step}: loss={metrics['loss']:.4f}, IoU={metrics['mean_iou']:.4f}", flush=True)
if curve[-1]["loss"] >= 0.8 * curve[0]["loss"]:
    raise AssertionError("WHU tiny learning check did not reduce loss by 20%.")

metadata = build_metadata(ROOT, manifest, splits, imagenet_config, device)
with tempfile.TemporaryDirectory() as temporary_directory:
    checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
    torch.save({
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": 60,
        "config_sha256": metadata["compatibility_identity"]["config_sha256"],
        "input_sha256": metadata["compatibility_identity"]["input_sha256"],
    }, checkpoint_path)
    restored, _ = build_model("random", imagenet_config["seeds"]["model"])
    restored_step = restore_checkpoint(
        restored, checkpoint_path,
        metadata["compatibility_identity"]["config_sha256"],
        metadata["compatibility_identity"]["input_sha256"],
    )
    if restored_step != 60 or state_digest(restored) != state_digest(model):
        raise AssertionError("Strict WHU checkpoint restoration changed the learned model.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "training_ids": random_config["training_ids"],
    "validation_image_count": len(random_config["validation_ids"]),
    "provider_test_excluded": True,
    "target_mapping": "building footprint = mask == 0",
    "empty_target_checked": empty_id,
    "paired_schedule_sha256": schedule_record["sha256"],
    "paired_decoder_sha256": random_initialization["decoder_sha256"],
    "random_encoder_sha256": random_initialization["encoder_sha256"],
    "imagenet_encoder_sha256": imagenet_initialization["encoder_sha256"],
    "pretrained_weights": imagenet_initialization["pretrained_weights"],
    "optimizer_groups": optimizer_record,
    "mps_forward_backward": {"output_shape": list(output.shape), "loss": values["loss"]},
    "tiny_learning": {"ids": positive_ids, "augmentation": False, "curve": curve,
                      "loss_reduction_fraction": 1 - curve[-1]["loss"] / curve[0]["loss"]},
    "checkpoint_restoration": True,
}
(ROOT / "reports/whu_training_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"device": report["device"], "loss_reduction_fraction": report["tiny_learning"]["loss_reduction_fraction"]}, indent=2))
