"""Focused aggregation tests for the Stage 3 training-horizon report."""

import pandas as pd

from roofseg.horizon_analysis import summarize_horizon_history


def test_horizon_summary_separates_2000_and_4000_step_selection():
    history = pd.DataFrame({
        "step": [0, 100, 2000, 2100, 4000],
        "validation_mean_iou": [0.1, 0.5, 0.7, 0.8, 0.75],
    })
    result = summarize_horizon_history(history)
    assert result == {
        "best_step_first_2000": 2000,
        "best_iou_first_2000": 0.7,
        "best_step_all_4000": 2100,
        "best_iou_all_4000": 0.8,
        "iou_step_2000": 0.7,
        "iou_step_4000": 0.75,
    }
