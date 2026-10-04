"""Focused aggregation tests for the Stage 3 training-horizon report."""

import pandas as pd
import pytest

from roofseg.horizon_analysis import summarize_horizon_history
from roofseg.late_training_analysis import summarize_late_history


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


def test_late_trend_summary_uses_fixed_windows_and_residuals():
    steps = list(range(2100, 4001, 100))
    history = pd.DataFrame({
        "step": steps,
        "validation_mean_iou": [0.5 + 0.00001 * step for step in steps],
        "validation_loss": [0.4 - 0.00002 * step for step in steps],
    })
    summary, rolling = summarize_late_history(history)
    assert summary["iou_trend_per_1000"] == pytest.approx(0.01)
    assert summary["loss_trend_per_1000"] == pytest.approx(-0.02)
    assert summary["iou_residual_sd"] == pytest.approx(0.0, abs=1e-12)
    assert summary["loss_residual_sd"] == pytest.approx(0.0, abs=1e-12)
    assert len(rolling) == 11
    assert rolling.iloc[0].window_start_step == 2100
    assert rolling.iloc[-1].window_end_step == 4000
    assert rolling.iou_trend_per_1000.to_list() == pytest.approx([0.01] * 11)
