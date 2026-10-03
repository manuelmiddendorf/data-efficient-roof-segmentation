import torch

from roofseg.training_data import apply_d4, paired_schedule


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
