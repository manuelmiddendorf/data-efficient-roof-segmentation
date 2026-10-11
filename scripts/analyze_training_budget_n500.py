"""Create tables and figures for the n=500 training-budget comparison."""

from pathlib import Path

from roofseg.training_budget import (
    collect_training_budget_results,
    plot_training_budget_histories,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
results, paired, history = collect_training_budget_results(ROOT)
results.to_csv(ROOT / "reports/training_budget_n500_results.csv", index=False)
paired.to_csv(ROOT / "reports/training_budget_n500_paired.csv", index=False)
history.to_csv(ROOT / "reports/training_budget_n500_history.csv", index=False)
figure = plot_training_budget_histories(history)
figure.savefig(
    FIGURES / "training_budget_n500_histories.png", dpi=170, bbox_inches="tight"
)
print("Training-budget results:\n", results.to_string(index=False))
print("\nPaired ImageNet-minus-random differences:\n", paired.to_string(index=False))
