"""Create compact tables and figures for the WHU training-budget block."""

from pathlib import Path

from roofseg.whu_training_budget import (
    collect_whu_training_budget_results,
    plot_whu_training_budget_histories,
    rid_whu_budget_pattern_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
results, paired, summary, history = collect_whu_training_budget_results(ROOT)
comparison = rid_whu_budget_pattern_comparison(ROOT, results, paired)
results.to_csv(ROOT / "reports/whu_training_budget_results.csv", index=False)
paired.to_csv(ROOT / "reports/whu_training_budget_paired.csv", index=False)
summary.to_csv(ROOT / "reports/whu_training_budget_summary.csv", index=False)
history.to_csv(ROOT / "reports/whu_training_budget_history.csv", index=False)
comparison.to_csv(
    ROOT / "reports/whu_rid_training_budget_comparison.csv", index=False
)
figure = plot_whu_training_budget_histories(history)
figure.savefig(
    FIGURES / "07_whu_training_budget_histories.png",
    dpi=170,
    bbox_inches="tight",
)
print("WHU budget results:\n", results.to_string(index=False))
print("\nPaired ImageNet-minus-Random gaps:\n", paired.to_string(index=False))
print("\nDescriptive budget-change summary:\n", summary.to_string(index=False))
print("\nRID/WHU pattern comparison:\n", comparison.to_string(index=False))
