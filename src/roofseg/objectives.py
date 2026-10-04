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
