"""Tensor loading, ImageNet normalization and paired D4 augmentation for RID."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import load_binary_target, read_image, read_mask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RIDTensorStore:
    """Load RID samples according to the saved manifest.

    Images become float32 ``3×512×512`` tensors normalized with the ImageNet-1K
    EfficientNet-B0 statistics. Targets become float32 ``1×512×512`` tensors
    with roof codes 0--16 mapped to one and background code 17 mapped to zero.
    """

    def __init__(self, project_root: Path, manifest: dict):
        self.project_root = project_root
        self.records = {sample["id"]: sample for sample in manifest["samples"]}
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

    def load(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one normalized image and aligned binary roof target."""
        if sample_id not in self.records:
            raise KeyError(f"RID sample is absent from the manifest: {sample_id}")
        record = self.records[sample_id]
        image = torch.from_numpy(read_image(self.project_root / record["image"]).copy())
        image = image.permute(2, 0, 1).float().div_(255.0)
        image = (image - self.mean) / self.std
        target_array = load_binary_target(read_mask(self.project_root / record["mask"]))
        target = torch.from_numpy(target_array.copy()).unsqueeze(0).float()
        return image, target

    def batch(self, sample_ids: Sequence[str], d4_codes: Sequence[int] | None = None) -> dict:
        """Load a batch and optionally apply aligned D4 transforms.

        D4 codes 0--3 are rotations by multiples of 90 degrees. Codes 4--7
        apply a horizontal reflection before the corresponding rotation.
        """
        loaded = [self.load(sample_id) for sample_id in sample_ids]
        images = torch.stack([item[0] for item in loaded])
        targets = torch.stack([item[1] for item in loaded])
        if d4_codes is not None:
            if len(d4_codes) != len(sample_ids):
                raise ValueError("Each sample requires one D4 code.")
            transformed = [apply_d4(images[index], targets[index], int(code))
                           for index, code in enumerate(d4_codes)]
            images = torch.stack([item[0] for item in transformed])
            targets = torch.stack([item[1] for item in transformed])
        return {"id": list(sample_ids), "image": images, "target": targets}


def apply_d4(image: torch.Tensor, target: torch.Tensor, code: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of eight square symmetries identically to image and target."""
    if image.shape != (3, 512, 512) or target.shape != (1, 512, 512) or code not in range(8):
        raise ValueError("Expected 3×512×512 image, 1×512×512 target and D4 code 0--7.")
    if code >= 4:
        image, target = image.flip(-1), target.flip(-1)
    turns = code % 4
    return torch.rot90(image, turns, (-2, -1)), torch.rot90(target, turns, (-2, -1))


def paired_schedule(
    sample_ids: Sequence[str], max_steps: int, batch_size: int, order_seed: int, augmentation_seed: int
) -> tuple[list[tuple[list[str], list[int]]], dict]:
    """Create a deterministic shuffled batch and augmentation schedule.

    The last incomplete batch of every data passage is retained. The schedule is
    created before model construction, so encoder weight loading cannot alter it.
    """
    if len(sample_ids) < 1 or max_steps < 1 or batch_size < 1:
        raise ValueError("Training schedule inputs must be positive and nonempty.")
    ids = np.asarray(list(sample_ids), dtype=object)
    order_rng = np.random.default_rng(order_seed)
    augmentation_rng = np.random.default_rng(augmentation_seed)
    schedule: list[tuple[list[str], list[int]]] = []
    passages = 0
    while len(schedule) < max_steps:
        passages += 1
        permutation = ids[order_rng.permutation(len(ids))]
        for start in range(0, len(ids), batch_size):
            batch_ids = permutation[start:start + batch_size].tolist()
            codes = augmentation_rng.integers(0, 8, size=len(batch_ids)).astype(int).tolist()
            schedule.append((batch_ids, codes))
            if len(schedule) == max_steps:
                break
    serialized = json.dumps(schedule, separators=(",", ":")).encode()
    examples = sum(len(batch_ids) for batch_ids, _ in schedule)
    return schedule, {
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "optimizer_steps": max_steps,
        "examples_processed": examples,
        "equivalent_data_passages": examples / len(ids),
        "completed_or_partial_passages": passages,
        "incomplete_batches_retained": len(ids) % batch_size != 0,
    }
