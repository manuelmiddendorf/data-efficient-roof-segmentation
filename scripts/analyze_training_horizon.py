"""Create compact Stage 3 training-horizon tables and validation figures."""

from pathlib import Path

import torch

from roofseg.horizon_analysis import (
    choose_horizon_examples,
    collect_horizon_tables,
    plot_horizon_curves,
    plot_horizon_examples,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, history, metrics = collect_horizon_tables(ROOT)
selected_ids = choose_horizon_examples(results, metrics)

results.to_csv(ROOT / "reports/optimization_training_horizon_results.csv", index=False)
history.to_csv(ROOT / "reports/optimization_training_horizon_history.csv", index=False)
(ROOT / "reports/optimization_training_horizon_example_ids.txt").write_text(
    "\n".join(selected_ids) + "\n", encoding="utf-8"
)

figures = {
    "optimization_training_horizon_curves.png": plot_horizon_curves(history),
    "optimization_training_horizon_examples.png": plot_horizon_examples(
        ROOT, results, selected_ids
    ),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

print(results.to_string(index=False))
print("\nSelected validation IDs:", ", ".join(selected_ids))
