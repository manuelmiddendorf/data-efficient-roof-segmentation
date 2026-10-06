"""Binary roof loss and equally weighted per-image segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    """Average equal-weight BCE-with-logits and soft Dice over images.

    BCE is first averaged over all pixels of each image, then over images. Soft
    Dice is computed per image with additive ``epsilon`` in numerator and
    denominator, then averaged. An empty target is therefore still penalized for
    predicted roof probability; an empty target and vanishing prediction tends
    to Dice one.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Loss expects matching N×1×H×W logits and targets.")
    axes = (1, 2, 3)
    bce_per_image = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(axes)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(axes)
    denominator = (probability + target).sum(axes)
    dice_per_image = (2 * intersection + epsilon) / (denominator + epsilon)
    bce = bce_per_image.mean()
    soft_dice_loss = (1 - dice_per_image).mean()
    return 0.5 * bce + 0.5 * soft_dice_loss, {"bce": bce, "soft_dice_loss": soft_dice_loss}


def boundary_band(target: torch.Tensor, radius: int = 3, inner: bool = False) -> torch.Tensor:
    """Construct a binary roof-boundary band while excluding the crop border.

    Parameters
    ----------
    target:
        Binary tensor with shape ``N×1×H×W``.
    radius:
        Pixel radius. Radius three uses a square ``7×7`` structuring element.
    inner:
        If false, return dilation minus erosion. If true, return target minus
        erosion, as used for the symmetric Boundary-IoU diagnostic.

    Returns
    -------
    torch.Tensor
        Float tensor aligned with ``target``. The outer ``radius`` rows and
        columns are zero so the image crop cannot create a measured boundary.
    """
    if target.ndim != 4 or target.shape[1] != 1 or radius < 1:
        raise ValueError("Boundary bands expect N×1×H×W targets and positive radius.")
    if target.shape[-2] <= 2 * radius or target.shape[-1] <= 2 * radius:
        raise ValueError("Boundary radius leaves no interior pixels.")
    if not torch.all((target == 0) | (target == 1)).item():
        raise ValueError("Boundary bands require binary targets.")
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=radius)
    eroded = 1 - F.max_pool2d(1 - target, kernel_size, stride=1, padding=radius)
    band = target - eroded if inner else dilated - eroded
    band = (band > 0).to(target.dtype)
    interior = torch.zeros_like(band)
    interior[..., radius:-radius, radius:-radius] = 1
    return band * interior


def boundary_weighted_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    normalization: str,
    mean_training_band_fraction: float,
    radius: int = 3,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add one of the two fixed boundary-weighted BCE terms to the base loss.

    ``proportional_band_area`` adds each image's band BCE divided by all image
    pixels. ``equal_per_nonempty_band`` multiplies mean BCE within each nonempty
    band by the fixed mean training-band fraction. Both variants are averaged
    equally over images; empty bands contribute zero.
    """
    base_loss, base_components = segmentation_loss(logits, target, epsilon)
    if normalization not in {"proportional_band_area", "equal_per_nonempty_band"}:
        raise ValueError(f"Unknown boundary normalization: {normalization}")
    if not 0 <= mean_training_band_fraction <= 1:
        raise ValueError("Mean training-band fraction must lie in [0, 1].")
    band = boundary_band(target, radius=radius)
    bce_pixels = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    axes = (1, 2, 3)
    band_pixels = band.sum(axes)
    band_bce_sum = (bce_pixels * band).sum(axes)
    nonempty = band_pixels > 0
    band_bce_per_image = torch.where(
        nonempty,
        band_bce_sum / band_pixels.clamp_min(1),
        torch.zeros_like(band_bce_sum),
    )
    band_fraction = band_pixels / target[0].numel()
    coefficient = (
        band_fraction
        if normalization == "proportional_band_area"
        else torch.full_like(band_fraction, mean_training_band_fraction) * nonempty
    )
    boundary_addition = (coefficient * band_bce_per_image).mean()
    components = {
        **base_components,
        "base_loss": base_loss,
        "boundary_bce": band_bce_per_image.mean(),
        "boundary_coefficient": coefficient.mean(),
        "boundary_addition": boundary_addition,
    }
    return base_loss + boundary_addition, components


def boundary_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    radius: int = 3,
) -> float | None:
    """Compute symmetric IoU between inner prediction and reference bands.

    The outer ``radius`` pixels are excluded by :func:`boundary_band`. ``None``
    denotes a reference without an observable inner contour and allows callers
    to define one common evaluation subset from reference masks alone. A missing
    predicted contour for an included reference returns zero.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Boundary IoU expects aligned two-dimensional masks.")
    tensors = torch.from_numpy(np.stack([prediction, target]).astype(np.float32))[:, None]
    bands = boundary_band(tensors, radius=radius, inner=True).numpy()[:, 0].astype(bool)
    predicted_band, reference_band = bands
    if not reference_band.any():
        return None
    union = np.logical_or(predicted_band, reference_band).sum()
    return float(np.logical_and(predicted_band, reference_band).sum() / union) if union else 0.0


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute roof IoU and Dice for one image with explicit empty-set rules."""
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError("Metrics expect aligned two-dimensional arrays.")
    prediction = probability >= threshold
    truth = target.astype(bool)
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "iou": true_positive / union if union else 1.0,
        "dice": 2 * true_positive / dice_denominator if dice_denominator else 1.0,
    }


def mean_metrics_at_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    thresholds: tuple[float, ...],
) -> list[dict]:
    """Compute equally weighted mean per-image IoU and Dice at fixed thresholds.

    Parameters
    ----------
    probability:
        Roof probabilities with shape ``(N, H, W)`` and values in ``[0, 1]``.
    target:
        Aligned binary targets with shape ``(N, H, W)``.
    thresholds:
        Non-empty threshold sequence with values strictly between zero and one.

    Returns
    -------
    list of dict
        Threshold, mean IoU, mean Dice and image count. Empty images use the
        same conventions as :func:`binary_metrics`.
    """
    if probability.shape != target.shape or probability.ndim != 3:
        raise ValueError("Threshold metrics expect aligned N×H×W arrays.")
    if not thresholds or any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("Thresholds must be non-empty and strictly between zero and one.")
    rows = []
    for threshold in thresholds:
        per_image = [
            binary_metrics(probability[index], target[index], threshold)
            for index in range(len(probability))
        ]
        rows.append({
            "threshold": threshold,
            "mean_iou": float(np.mean([row["iou"] for row in per_image])),
            "mean_dice": float(np.mean([row["dice"] for row in per_image])),
            "image_count": len(per_image),
        })
    return rows


def average_precision(probability: np.ndarray, target: np.ndarray) -> float | None:
    """Compute non-interpolated pixel AP for one image, or ``None`` if empty."""
    scores = probability.ravel().astype(np.float64)
    truth = target.ravel().astype(bool)
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, truth = scores[order], truth[order]
    cumulative_true = np.cumsum(truth)
    threshold_ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    true_at_threshold = cumulative_true[threshold_ends]
    precision = true_at_threshold / (threshold_ends + 1)
    recall = true_at_threshold / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
