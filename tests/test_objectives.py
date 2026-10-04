import numpy as np
import torch

from roofseg.objectives import (
    average_precision,
    binary_metrics,
    mean_metrics_at_thresholds,
    segmentation_loss,
)


def test_loss_is_finite_and_rewards_correct_logits_including_empty_target():
    target = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    wrong = torch.zeros_like(target)
    correct = torch.tensor([[[[8.0, -8.0]]], [[[-8.0, -8.0]]]])
    wrong_loss, _ = segmentation_loss(wrong, target)
    correct_loss, _ = segmentation_loss(correct, target)
    assert torch.isfinite(correct_loss)
    assert correct_loss < wrong_loss


def test_metrics_define_empty_masks_and_false_positives():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert binary_metrics(np.zeros((2, 2)), empty) == {
        "tp": 0, "fp": 0, "fn": 0, "iou": 1.0, "dice": 1.0
    }
    assert binary_metrics(np.ones((2, 2)), empty)["iou"] == 0.0
    assert average_precision(np.zeros((2, 2)), empty) is None


def test_threshold_sweep_preserves_equal_image_weight_and_empty_rules():
    probability = np.array([
        [[0.2, 0.1], [0.1, 0.2]],
        [[0.8, 0.4], [0.7, 0.1]],
    ])
    target = np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [1, 0]],
    ])
    rows = mean_metrics_at_thresholds(probability, target, (0.1, 0.5, 0.9))
    assert rows[1] == {"threshold": 0.5, "mean_iou": 1.0,
                       "mean_dice": 1.0, "image_count": 2}
    assert rows[0]["mean_iou"] == 0.25
    assert rows[2]["mean_iou"] == 0.5


def test_average_precision_uses_ranked_positive_pixels():
    probability = np.array([[0.9, 0.8, 0.7, 0.1]])
    target = np.array([[1, 0, 1, 0]])
    assert np.isclose(average_precision(probability, target), (1.0 + 2 / 3) / 2)
