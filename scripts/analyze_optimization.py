"""Create compact Stage 3 learning-rate tables and validation figures."""

from pathlib import Path

import torch

from roofseg.optimization_analysis import (
    choose_optimization_examples,
    collect_optimization_tables,
    evaluate_reference_thresholds,
    plot_learning_rate_curves,
    plot_optimization_examples,
    plot_paired_learning_rate_differences,
    plot_threshold_curves,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, history, metrics = collect_optimization_tables(ROOT)
thresholds = evaluate_reference_thresholds(ROOT)
selected_ids = choose_optimization_examples(results, metrics)

results.to_csv(ROOT / "reports/optimization_learning_rate_results.csv", index=False)
history.to_csv(ROOT / "reports/optimization_learning_rate_history.csv", index=False)
thresholds.to_csv(ROOT / "reports/optimization_threshold_results.csv", index=False)
(ROOT / "reports/optimization_example_ids.txt").write_text(
    "\n".join(selected_ids) + "\n", encoding="utf-8"
)

figures = {
    "optimization_learning_rate_curves.png": plot_learning_rate_curves(history),
    "optimization_paired_differences.png": plot_paired_learning_rate_differences(results),
    "optimization_threshold_curves.png": plot_threshold_curves(thresholds),
    "optimization_examples.png": plot_optimization_examples(ROOT, results, selected_ids),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

print(results.to_string(index=False))
print("\nThreshold analysis:\n", thresholds.to_string(index=False))
print("\nSelected validation IDs:", ", ".join(selected_ids))
