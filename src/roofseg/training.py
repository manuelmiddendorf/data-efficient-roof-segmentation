"""Step-based training and evaluation for the fixed Stage 2 pilot."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .integrity import sha256_file
from .model import build_model, state_digest
from .objectives import average_precision, binary_metrics, segmentation_loss
from .run_artifacts import write_csv, write_json
from .training_data import RIDTensorStore, paired_schedule


def select_device() -> torch.device:
    """Select MPS when available, otherwise CPU, without runtime fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def adamw(model: nn.Module, learning_rate: float, weight_decay: float):
    """Create AdamW with zero decay for biases and one-dimensional norm parameters."""
    decay, no_decay, decay_names, no_decay_names = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )
    return optimizer, {
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "weight_decays": [group["weight_decay"] for group in optimizer.param_groups],
        "decayed_parameter_tensors": len(decay_names),
        "no_decay_parameter_tensors": len(no_decay_names),
        "no_decay_rule": "parameter.ndim == 1 or name ends with .bias",
    }


def train_step(model, optimizer, batch: dict, device: torch.device) -> dict:
    """Run one float32 optimizer step and return scalar objective components."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    image, target = batch["image"].to(device), batch["target"].to(device)
    logits = model(image)
    loss, components = segmentation_loss(logits, target)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Training loss became non-finite.")
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), **{name: value.item() for name, value in components.items()}}


def evaluate(
    model: nn.Module,
    store: RIDTensorStore,
    sample_ids: list[str],
    batch_size: int,
    device: torch.device,
    include_ap: bool = False,
) -> tuple[list[dict], dict]:
    """Evaluate original-orientation images with equal per-image aggregation."""
    model.eval()
    rows, weighted_loss = [], {"loss": 0.0, "bce": 0.0, "soft_dice_loss": 0.0}
    with torch.inference_mode():
        for start in range(0, len(sample_ids), batch_size):
            batch = store.batch(sample_ids[start:start + batch_size])
            image, target = batch["image"].to(device), batch["target"].to(device)
            logits = model(image)
            loss, components = segmentation_loss(logits, target)
            probabilities = logits.sigmoid().cpu().numpy()[:, 0]
            targets = batch["target"].numpy()[:, 0]
            for index, sample_id in enumerate(batch["id"]):
                metrics = binary_metrics(probabilities[index], targets[index])
                row = {"id": sample_id, **metrics}
                if include_ap:
                    row["average_precision"] = average_precision(probabilities[index], targets[index])
                rows.append(row)
            count = len(batch["id"])
            for name, value in {"loss": loss, **components}.items():
                weighted_loss[name] += value.item() * count
    summary = {name: value / len(sample_ids) for name, value in weighted_loss.items()}
    summary.update({
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
    })
    if include_ap:
        aps = [row["average_precision"] for row in rows if row["average_precision"] is not None]
        summary.update({"mean_average_precision": float(np.mean(aps)), "ap_image_count": len(aps)})
    return rows, summary


def fit_run(
    project_root: Path, config: dict, metadata: dict, manifest: dict,
    device: torch.device, directory: Path,
) -> dict:
    """Train one resolved run, select by validation IoU, and save six artifacts."""
    store = RIDTensorStore(project_root, manifest)
    schedule, schedule_record = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    if schedule_record["sha256"] != metadata["compatibility_identity"]["schedule_sha256"]:
        raise AssertionError("Training schedule differs from the precomputed identity.")
    model, initialization = build_model(config["initialization"], config["seeds"]["model"])
    metadata["weight_source"] = (
        initialization["pretrained_weights"]
        if initialization["pretrained_weights"] is not None
        else {"kind": "fresh random encoder initialization", "external_weights": False}
    )
    metadata["initial_state"] = {
        "encoder_sha256": initialization["encoder_sha256"],
        "decoder_sha256": initialization["decoder_sha256"],
    }
    write_json(directory / "metadata.json", metadata)
    model.to(device)
    optimizer, optimizer_record = adamw(model, config["learning_rate"], config["weight_decay"])
    history: list[dict] = []
    best_iou, best_step = -math.inf, None
    training_seconds, evaluation_seconds, examples_processed = 0.0, 0.0, 0
    started = time.perf_counter()

    before_rows, before = evaluate(model, store, config["validation_ids"], config["batch_size"], device)
    history.append({"step": 0, "examples_processed": 0, "data_passages": 0.0,
                    "train_loss": "", "train_bce": "", "train_soft_dice_loss": "",
                    **{f"validation_{name}": value for name, value in before.items()},
                    "elapsed_seconds": time.perf_counter() - started})
    interval = []
    for step, (batch_ids, d4_codes) in enumerate(schedule, 1):
        tick = time.perf_counter()
        values = train_step(model, optimizer, store.batch(batch_ids, d4_codes), device)
        synchronize(device)
        training_seconds += time.perf_counter() - tick
        examples_processed += len(batch_ids)
        interval.append(values)
        if step % config["evaluation_interval"] != 0:
            continue
        tick = time.perf_counter()
        validation_rows, validation = evaluate(
            model, store, config["validation_ids"], config["batch_size"], device
        )
        synchronize(device)
        evaluation_seconds += time.perf_counter() - tick
        means = {name: float(np.mean([row[name] for row in interval])) for name in interval[0]}
        row = {"step": step, "examples_processed": examples_processed,
               "data_passages": examples_processed / len(config["training_ids"]),
               **{f"train_{name}": value for name, value in means.items()},
               **{f"validation_{name}": value for name, value in validation.items()},
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        interval = []
        write_csv(directory / "history.csv", history)
        print(f"{directory.name}: step {step}/{config['max_steps']} "
              f"loss={means['loss']:.4f} val_iou={validation['mean_iou']:.4f}", flush=True)
        if validation["mean_iou"] > best_iou:
            best_iou, best_step = validation["mean_iou"], step
            torch.save({
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "step": step,
                "config_sha256": metadata["compatibility_identity"]["config_sha256"],
                "input_sha256": metadata["compatibility_identity"]["input_sha256"],
            }, directory / "checkpoint.pt")

    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    training_rows, training_metrics = evaluate(
        model, store, config["training_ids"], config["batch_size"], device, include_ap=True
    )
    validation_rows, validation_metrics = evaluate(
        model, store, config["validation_ids"], config["batch_size"], device, include_ap=True
    )
    metrics_rows = ([{"split": "training", **row} for row in training_rows]
                    + [{"split": "validation", **row} for row in validation_rows])
    write_csv(directory / "metrics.csv", metrics_rows)
    summary = {
        "status": "completed",
        "selection": "highest mean per-image validation IoU at threshold 0.5; earlier step wins ties",
        "best_step": best_step,
        "best_validation": validation_metrics,
        "matched_training_evaluation": training_metrics,
        "initial_validation": before,
        "steps_completed": config["max_steps"],
        "examples_processed": schedule_record["examples_processed"],
        "equivalent_data_passages": schedule_record["equivalent_data_passages"],
        "incomplete_batches_retained": schedule_record["incomplete_batches_retained"],
        "training_seconds": training_seconds,
        "evaluation_seconds_during_training": evaluation_seconds,
        "total_seconds": time.perf_counter() - started,
        "initialization": initialization,
        "optimizer_groups": optimizer_record,
        "best_model_sha256": state_digest(model),
    }
    summary["checkpoint_sha256"] = sha256_file(directory / "checkpoint.pt")
    write_json(directory / "summary.json", summary)
    return summary


def restore_checkpoint(model: nn.Module, checkpoint_path: Path, config_sha256: str, input_sha256: str) -> int:
    """Strictly restore a checkpoint after configuration and input identity checks."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != config_sha256 or checkpoint["input_sha256"] != input_sha256:
        raise ValueError("Checkpoint configuration or input identity mismatch.")
    model.load_state_dict(checkpoint["model"], strict=True)
    return int(checkpoint["step"])
