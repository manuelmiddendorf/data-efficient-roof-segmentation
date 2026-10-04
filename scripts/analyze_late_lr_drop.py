"""Create late-trajectory diagnostics and fixed-drop comparison artifacts."""

from pathlib import Path

from roofseg.late_training_analysis import (
    collect_constant_late_trends,
    collect_lr_drop_comparison,
    plot_late_trends,
    plot_lr_drop_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

trends, rolling, constant_history = collect_constant_late_trends(ROOT)
comparison, comparison_history, comparison_rolling = collect_lr_drop_comparison(ROOT)

trends.to_csv(ROOT / "reports/optimization_late_training_trends.csv", index=False)
rolling.to_csv(ROOT / "reports/optimization_late_training_rolling_trends.csv", index=False)
comparison.to_csv(ROOT / "reports/optimization_lr_drop_results.csv", index=False)
comparison_history.to_csv(ROOT / "reports/optimization_lr_drop_history.csv", index=False)
comparison_rolling.to_csv(ROOT / "reports/optimization_lr_drop_rolling_trends.csv", index=False)

figures = {
    "optimization_late_training_trends.png": plot_late_trends(constant_history, rolling),
    "optimization_lr_drop_curves.png": plot_lr_drop_comparison(comparison_history),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

print("Late constant-rate trends:\n", trends.to_string(index=False))
print("\nFixed-drop comparison:\n", comparison.to_string(index=False))
