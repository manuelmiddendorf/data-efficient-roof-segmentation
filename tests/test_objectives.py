import numpy as np
import torch

from roofseg.objectives import (
    average_precision,
    binary_metrics,
    boundary_band,
    boundary_iou,
    boundary_weighted_segmentation_loss,
    mean_metrics_at_thresholds,
    segmentation_loss,
)
from roofseg.training_data import apply_d4


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


def test_boundary_band_geometry_border_exclusion_and_empty_masks():
    target = torch.zeros(1, 1, 9, 9)
    target[..., 4, 4] = 1
    band = boundary_band(target, radius=1)
    assert band.sum().item() == 9
    assert band[..., 3:6, 3:6].all()

    crop_edge_target = torch.zeros_like(target)
    crop_edge_target[..., 2:7, :3] = 1
    crop_edge_band = boundary_band(crop_edge_target, radius=1)
    assert not crop_edge_band[..., 0, :].any()
    assert not crop_edge_band[..., -1, :].any()
    assert not crop_edge_band[..., :, 0].any()
    assert not crop_edge_band[..., :, -1].any()
    assert boundary_band(torch.zeros_like(target), radius=1).sum().item() == 0
    assert boundary_band(torch.ones_like(target), radius=1).sum().item() == 0


def test_boundary_band_commutes_with_d4_square_symmetries():
    image = torch.zeros(3, 512, 512)
    target = torch.zeros(1, 512, 512)
    target[:, 40:160, 80:230] = 1
    original_band = boundary_band(target.unsqueeze(0))[0]
    for code in range(8):
        _, transformed_target = apply_d4(image, target, code)
        _, transformed_band = apply_d4(image, original_band, code)
        torch.testing.assert_close(
            boundary_band(transformed_target.unsqueeze(0))[0], transformed_band
        )


def test_boundary_weighted_formulas_match_analytic_constant_bce():
    target = torch.zeros(2, 1, 9, 9)
    target[0, :, 3:6, 3:6] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    base, _ = segmentation_loss(logits, target)
    band = boundary_band(target, radius=1)
    fractions = band.sum((1, 2, 3)) / 81
    constant_bce = torch.log(torch.tensor(2.0))
    mean_fraction = 0.2

    area_loss, area_components = boundary_weighted_segmentation_loss(
        logits, target, "proportional_band_area", mean_fraction, radius=1
    )
    expected_area_addition = (fractions[0] * constant_bce) / 2
    torch.testing.assert_close(area_loss, base + expected_area_addition)
    torch.testing.assert_close(area_components["boundary_addition"], expected_area_addition)

    equal_loss, equal_components = boundary_weighted_segmentation_loss(
        logits, target, "equal_per_nonempty_band", mean_fraction, radius=1
    )
    expected_equal_addition = mean_fraction * constant_bce / 2
    torch.testing.assert_close(equal_loss, base + expected_equal_addition)
    torch.testing.assert_close(equal_components["boundary_addition"], expected_equal_addition)
    equal_loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_boundary_iou_is_symmetric_and_handles_missing_contours():
    target = np.zeros((11, 11), dtype=bool)
    target[3:8, 3:8] = True
    assert boundary_iou(target, target, radius=1) == 1.0
    assert boundary_iou(np.zeros_like(target), target, radius=1) == 0.0
    shifted = np.zeros_like(target)
    shifted[3:8, 4:9] = True
    forward = boundary_iou(shifted, target, radius=1)
    reverse = boundary_iou(target, shifted, radius=1)
    assert forward == reverse
    assert 0 < forward < 1
    assert boundary_iou(np.zeros_like(target), np.zeros_like(target), radius=1) is None
