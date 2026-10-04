"""Create compact tables and figures from completed active Stage 2 runs."""

from pathlib import Path

import torch

from roofseg.pilot import PILOT_SIZES
from roofseg.pilot_analysis import (choose_validation_examples, collect_pilot_tables,
                                    plot_learning_curves, plot_paired_differences,
                                    plot_train_validation_iou, plot_validation_predictions)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, history, metrics = collect_pilot_tables(ROOT)
results.to_csv(ROOT / "reports/training_pilot_results.csv", index=False)
history.to_csv(ROOT / "reports/training_pilot_history.csv", index=False)

figures = {
    "training_pilot_learning_curves.png": plot_learning_curves(history),
    "training_pilot_paired_differences.png": plot_paired_differences(results),
    "training_pilot_train_validation.png": plot_train_validation_iou(results),
}
selected_ids = choose_validation_examples(metrics)
for size in PILOT_SIZES:
    figures[f"training_pilot_examples_n{size}.png"] = plot_validation_predictions(ROOT, size, selected_ids)
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

(ROOT / "reports/training_pilot_example_ids.txt").write_text(
    "\n".join(selected_ids) + "\n", encoding="utf-8"
)
print(results.to_string(index=False))
print("Selected validation IDs:", ", ".join(selected_ids))
