"""Run the real Stage 2 MPS smoke, pairing, timing and tiny-learning checks."""

from pathlib import Path
import json
import statistics
import tempfile
import time

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model, state_digest
from roofseg.objectives import segmentation_loss
from roofseg.pilot import build_config
from roofseg.training import adamw, evaluate, restore_checkpoint, select_device, synchronize, train_step
from roofseg.training_data import IMAGENET_MEAN, IMAGENET_STD, RIDTensorStore, paired_schedule
from roofseg.data import read_image, read_mask


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
config = build_config(splits, 25, "random")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Stage 2 pilot requires available MPS on this host; selected {device}.")

schedule_random, schedule_record = paired_schedule(
    config["training_ids"], 12, config["batch_size"],
    config["seeds"]["data_order"], config["seeds"]["augmentation"],
)
schedule_pretrained, _ = paired_schedule(
    config["training_ids"], 12, config["batch_size"],
    config["seeds"]["data_order"], config["seeds"]["augmentation"],
)
if schedule_random != schedule_pretrained:
    raise AssertionError("Paired schedules differ before model construction.")

store = RIDTensorStore(ROOT, manifest)
sample_id = config["training_ids"][0]
sample_record = store.records[sample_id]
raw_image = read_image(ROOT / sample_record["image"])
raw_mask = read_mask(ROOT / sample_record["mask"])
normalized_image, binary_target = store.load(sample_id)
expected_pixel = (
    raw_image[0, 0].astype("float32") / 255.0
    - torch.tensor(IMAGENET_MEAN).numpy()
) / torch.tensor(IMAGENET_STD).numpy()
torch.testing.assert_close(normalized_image[:, 0, 0], torch.from_numpy(expected_pixel))
if not torch.equal(binary_target[0], torch.from_numpy(raw_mask != 17)):
    raise AssertionError("RID target is not exactly mask != 17.")
if set(config["training_ids"]) & set(splits["roles"]["test"]):
    raise AssertionError("Training subset intersects the locked test role.")
if set(config["validation_ids"]) & set(splits["roles"]["test"]):
    raise AssertionError("Validation role intersects the locked test role.")

random_model, random_initialization = build_model("random", config["seeds"]["model"])
pretrained_model, pretrained_initialization = build_model("imagenet", config["seeds"]["model"])
if random_initialization["decoder_sha256"] != pretrained_initialization["decoder_sha256"]:
    raise AssertionError("Paired decoder initializations differ.")
if random_initialization["encoder_sha256"] == pretrained_initialization["encoder_sha256"]:
    raise AssertionError("Random and ImageNet encoders unexpectedly match.")
del random_model

model = pretrained_model.to(device)
optimizer, optimizer_record = adamw(model, config["learning_rate"], config["weight_decay"])
first_batch_norm = next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))
running_before = first_batch_norm.running_mean.detach().cpu().clone()
timings = []
for index, (sample_ids, codes) in enumerate(schedule_random[:7]):
    batch = store.batch(sample_ids, codes)
    synchronize(device)
    tick = time.perf_counter()
    values = train_step(model, optimizer, batch, device)
    synchronize(device)
    if index >= 2:
        timings.append(time.perf_counter() - tick)
if torch.equal(running_before, first_batch_norm.running_mean.detach().cpu()):
    raise AssertionError("BatchNorm running statistics did not update in training mode.")
if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
    raise FloatingPointError("Smoke-test gradients are non-finite.")

model.eval()
running_before_eval = first_batch_norm.running_mean.detach().cpu().clone()
with torch.inference_mode():
    output = model(store.batch(config["training_ids"][:4])["image"].to(device))
if output.shape != (4, 1, 512, 512) or output.dtype != torch.float32 or output.device.type != "mps":
    raise AssertionError("Real forward pass violates shape, precision or device contract.")
if not torch.equal(running_before_eval, first_batch_norm.running_mean.detach().cpu()):
    raise AssertionError("BatchNorm running statistics changed in evaluation mode.")

tiny_ids = config["training_ids"][:2]
model, _ = build_model("imagenet", config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, config["learning_rate"], config["weight_decay"])
tiny_batch = store.batch(tiny_ids)
curve = []
for step in range(61):
    if step:
        train_step(model, optimizer, tiny_batch, device)
    if step % 10 == 0:
        rows, metrics = evaluate(model, store, tiny_ids, 2, device)
        curve.append({"step": step, "loss": metrics["loss"], "mean_iou": metrics["mean_iou"]})
        print(f"Tiny learning step {step}: loss={metrics['loss']:.4f}, IoU={metrics['mean_iou']:.4f}", flush=True)
if curve[-1]["loss"] >= 0.8 * curve[0]["loss"]:
    raise AssertionError("Tiny no-augmentation learning check did not reduce loss by 20%.")

with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "checkpoint.pt"
    torch.save({"model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "step": 60, "config_sha256": "preflight", "input_sha256": "tiny"}, path)
    restored, _ = build_model("random", config["seeds"]["model"])
    restored_step = restore_checkpoint(restored, path, "preflight", "tiny")
    if restored_step != 60 or state_digest(restored) != state_digest(model):
        raise AssertionError("Strict checkpoint restoration changed the learned model.")

evaluation_ids = config["validation_ids"][:16]
tick = time.perf_counter()
evaluate(model, store, evaluation_ids, 4, device)
synchronize(device)
seconds_per_validation_image = (time.perf_counter() - tick) / len(evaluation_ids)
median_step_seconds = statistics.median(timings)
estimated_seconds_per_run = (
    2000 * median_step_seconds
    + 21 * len(config["validation_ids"]) * seconds_per_validation_image
    + (500 + len(config["validation_ids"])) * seconds_per_validation_image
)
report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "split_counts": splits["counts"],
    "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
    "paired_schedule_sha256": schedule_record["sha256"],
    "paired_decoder_sha256": random_initialization["decoder_sha256"],
    "random_encoder_sha256": random_initialization["encoder_sha256"],
    "pretrained_encoder_sha256": pretrained_initialization["encoder_sha256"],
    "pretrained_weights": pretrained_initialization["pretrained_weights"],
    "optimizer_groups": optimizer_record,
    "batchnorm_train_updates_and_eval_freezes": True,
    "finite_backward": True,
    "real_data_contract": {
        "sample_id": sample_id,
        "target_mapping": "mask != 17",
        "target_values": sorted(binary_target.unique().int().tolist()),
        "imagenet_normalization_checked_against_raw_pixel": True,
        "locked_test_role_absent": True,
    },
    "timed_step_seconds_after_warmup": timings,
    "median_step_seconds": median_step_seconds,
    "seconds_per_validation_image_sample": seconds_per_validation_image,
    "estimated_seconds_per_run_upper_size": estimated_seconds_per_run,
    "estimated_seconds_six_runs": 6 * estimated_seconds_per_run,
    "tiny_learning": {"ids": tiny_ids, "augmentation": False, "curve": curve,
                      "loss_reduction_fraction": 1 - curve[-1]["loss"] / curve[0]["loss"]},
    "checkpoint_restoration": True,
    "mps_memory": {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                   "driver_allocated_bytes": torch.mps.driver_allocated_memory()},
}
(ROOT / "reports/training_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("device", "median_step_seconds", "estimated_seconds_six_runs")}, indent=2))
