"""Create compact WHU pilot tables and figures from completed artifacts."""

from pathlib import Path

import torch

from roofseg.whu_pilot_analysis import (
    choose_whu_validation_examples, collect_whu_pilot_tables,
    plot_whu_learning_curves, plot_whu_validation_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, history, metrics = collect_whu_pilot_tables(ROOT)
examples = choose_whu_validation_examples(metrics)
results.to_csv(ROOT / "reports/whu_training_pilot_results.csv", index=False)
history.to_csv(ROOT / "reports/whu_training_pilot_history.csv", index=False)
examples.to_csv(ROOT / "reports/whu_training_pilot_examples.csv", index=False)

figures = {
    "07_whu_pilot_learning_curves.png": plot_whu_learning_curves(history),
    "07_whu_pilot_validation_examples.png": plot_whu_validation_predictions(ROOT, examples),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

print(results.to_string(index=False))
print("\nSelected common examples:\n", examples.to_string(index=False))
