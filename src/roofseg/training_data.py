"""Tensor loading, ImageNet normalization and paired D4 augmentation for RID."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import functional as tv_functional

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

    def load_raw(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one float32 RGB image in ``[0, 1]`` and its binary target."""
        if sample_id not in self.records:
            raise KeyError(f"RID sample is absent from the manifest: {sample_id}")
        record = self.records[sample_id]
        image = torch.from_numpy(read_image(self.project_root / record["image"]).copy())
        image = image.permute(2, 0, 1).float().div_(255.0)
        target_array = load_binary_target(read_mask(self.project_root / record["mask"]))
        target = torch.from_numpy(target_array.copy()).unsqueeze(0).float()
        return image, target

    def load(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one normalized image and aligned binary roof target."""
        image, target = self.load_raw(sample_id)
        return (image - self.mean) / self.std, target

    def batch(
        self,
        sample_ids: Sequence[str],
        d4_codes: Sequence[int] | None = None,
        photometric_factors: Sequence[tuple[float, float]] | None = None,
    ) -> dict:
        """Load a batch and optionally apply photometric and aligned D4 transforms.

        D4 codes 0--3 are rotations by multiples of 90 degrees. Codes 4--7
        apply a horizontal reflection before the corresponding rotation. Each
        photometric pair contains a brightness factor followed by a contrast
        factor. Photometric transforms operate on RGB in ``[0, 1]`` before
        ImageNet normalization and do not modify targets.
        """
        loaded = [self.load_raw(sample_id) for sample_id in sample_ids]
        if photometric_factors is not None:
            if len(photometric_factors) != len(sample_ids):
                raise ValueError("Each sample requires one brightness/contrast pair.")
            loaded = [
                (apply_brightness_contrast(image, *photometric_factors[index]), target)
                for index, (image, target) in enumerate(loaded)
            ]
        images = torch.stack([item[0] for item in loaded])
        targets = torch.stack([item[1] for item in loaded])
        if d4_codes is not None:
            if len(d4_codes) != len(sample_ids):
                raise ValueError("Each sample requires one D4 code.")
            transformed = [apply_d4(images[index], targets[index], int(code))
                           for index, code in enumerate(d4_codes)]
            images = torch.stack([item[0] for item in transformed])
            targets = torch.stack([item[1] for item in transformed])
        images = (images - self.mean) / self.std
        return {"id": list(sample_ids), "image": images, "target": targets}


def apply_brightness_contrast(
    image: torch.Tensor, brightness_factor: float, contrast_factor: float
) -> torch.Tensor:
    """Apply whole-image brightness then contrast to float RGB in ``[0, 1]``."""
    if image.shape != (3, 512, 512) or image.dtype != torch.float32:
        raise ValueError("Expected a float32 3×512×512 RGB image.")
    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError("Photometric input must lie in [0, 1].")
    if brightness_factor < 0.0 or contrast_factor < 0.0:
        raise ValueError("Brightness and contrast factors must be non-negative.")
    brightened = tv_functional.adjust_brightness(image, brightness_factor)
    return tv_functional.adjust_contrast(brightened, contrast_factor)


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


def photometric_schedule(
    training_schedule: Sequence[tuple[Sequence[str], Sequence[int]]],
    seed: int,
    lower: float = 0.85,
    upper: float = 1.15,
) -> tuple[list[list[tuple[float, float]]], dict]:
    """Draw deterministic brightness/contrast pairs for every sample occurrence."""
    if lower < 0.0 or upper < lower:
        raise ValueError("Photometric factor bounds must satisfy 0 <= lower <= upper.")
    rng = np.random.default_rng(seed)
    factors = [
        [tuple(map(float, pair)) for pair in rng.uniform(lower, upper, size=(len(ids), 2))]
        for ids, _ in training_schedule
    ]
    serialized = json.dumps(factors, separators=(",", ":")).encode()
    return factors, {
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "seed": seed,
        "distribution": "independent uniform",
        "lower": lower,
        "upper": upper,
        "draws_per_occurrence": 2,
        "occurrences": sum(len(batch) for batch in factors),
    }
