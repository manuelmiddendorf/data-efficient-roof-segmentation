"""Create fixed-recipe Seed 17 data-efficiency result artifacts."""

from pathlib import Path

from roofseg.data_efficiency import (
    collect_data_efficiency_results,
    plot_data_efficiency_histories,
    plot_data_efficiency_summary,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
results, paired, history = collect_data_efficiency_results(ROOT)
results.to_csv(ROOT / "reports/data_efficiency_results.csv", index=False)
paired.to_csv(ROOT / "reports/data_efficiency_paired.csv", index=False)
history.to_csv(ROOT / "reports/data_efficiency_history.csv", index=False)
figures = {
    "data_efficiency_summary.png": plot_data_efficiency_summary(results),
    "data_efficiency_histories.png": plot_data_efficiency_histories(history),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
print("Data-efficiency results:\n", results.to_string(index=False))
print("\nPaired ImageNet-minus-random differences:\n", paired.to_string(index=False))
