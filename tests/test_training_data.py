import torch
from torchvision.transforms import functional as tv_functional

from roofseg.training_data import (
    apply_brightness_contrast,
    apply_d4,
    paired_schedule,
    photometric_schedule,
)


def test_d4_transform_keeps_image_and_target_aligned():
    target = torch.zeros((1, 512, 512))
    target[:, 20:40, 70:90] = 1
    image = torch.cat((target, target * 2, target * 3), dim=0)
    for code in range(8):
        transformed_image, transformed_target = apply_d4(image, target, code)
        torch.testing.assert_close(transformed_image[0], transformed_target[0])
        assert transformed_image.is_contiguous() is False or transformed_image.shape == image.shape


def test_schedule_is_paired_deterministic_and_retains_partial_batches():
    first, first_record = paired_schedule([str(i) for i in range(5)], 4, 4, 17, 29)
    second, second_record = paired_schedule([str(i) for i in range(5)], 4, 4, 17, 29)
    assert first == second
    assert first_record == second_record
    assert [len(ids) for ids, _ in first] == [4, 1, 4, 1]
    assert first_record["examples_processed"] == 10
    assert first_record["incomplete_batches_retained"]


def test_brightness_then_contrast_preserves_shape_range_and_does_not_touch_target():
    image = torch.linspace(0, 1, 3 * 512 * 512).reshape(3, 512, 512)
    target = torch.randint(0, 2, (1, 512, 512), dtype=torch.float32)
    transformed = apply_brightness_contrast(image, 1.15, 0.85)
    expected = tv_functional.adjust_contrast(
        tv_functional.adjust_brightness(image, 1.15), 0.85
    )
    torch.testing.assert_close(transformed, expected)
    assert transformed.shape == image.shape
    assert transformed.dtype == torch.float32
    assert 0.0 <= transformed.min().item() <= transformed.max().item() <= 1.0
    torch.testing.assert_close(target, target.clone())


def test_photometric_schedule_is_separate_reproducible_and_occurrence_aligned():
    base, base_record = paired_schedule([str(i) for i in range(5)], 4, 4, 17, 29)
    first, first_record = photometric_schedule(base, 1703)
    second, second_record = photometric_schedule(base, 1703)
    assert first == second
    assert first_record == second_record
    assert [len(batch) for batch in first] == [4, 1, 4, 1]
    assert first_record["occurrences"] == 10
    assert all(0.85 <= factor <= 1.15 for batch in first for pair in batch for factor in pair)
    _, unchanged_record = paired_schedule([str(i) for i in range(5)], 4, 4, 17, 29)
    assert unchanged_record == base_record
