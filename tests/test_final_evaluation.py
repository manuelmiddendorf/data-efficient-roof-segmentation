import numpy as np
import torch
from roofseg.final_evaluation import summarize_images, evaluate_test
from roofseg.objectives import binary_metrics, average_precision


def row(probability, target):
    return {**binary_metrics(probability, target), 'average_precision': average_precision(probability, target)}


def test_absent_groups_and_equal_image_aggregation():
    target = np.ones((2, 2))
    good = row(target, target)
    bad = row(np.zeros((2, 2)), target)
    summary = summarize_images([good, bad], 4)
    assert summary['mean_iou'] == 0.5
    assert summary['empty_image_count'] == 0
    assert summary['empty_fraction_with_any_prediction'] is None
    empty = summarize_images([row(np.zeros((2, 2)), np.zeros((2, 2)))], 4)
    assert empty['mean_iou'] == 1
    assert empty['mean_average_precision'] is None
    assert empty['positive_mean_iou'] is None


def test_single_false_positive_in_empty_image():
    summary = summarize_images([row(np.array([[1., 0.], [0., 0.]]), np.zeros((2, 2)))], 4)
    assert summary['mean_iou'] == 0
    assert summary['empty_mean_foreground_fraction'] == 0.25
    assert summary['empty_fraction_with_any_prediction'] == 1


def test_inference_eval_mode_order_threshold_and_partial_batch():
    class Store:
        def batch(self, ids):
            return {'id': ids, 'image': torch.zeros(len(ids), 1, 2, 2),
                    'target': torch.ones(len(ids), 1, 2, 2)}
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
    before = model[0].running_mean.clone()
    rows = evaluate_test(model, Store(), list('abcde'), torch.device('cpu'))
    assert [r['id'] for r in rows] == list('abcde')
    assert all(r['iou'] == 1 for r in rows)  # sigmoid(0) == .5 is foreground
    assert not model.training
    assert torch.equal(before, model[0].running_mean)
