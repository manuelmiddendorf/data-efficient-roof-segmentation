"""Create three-repetition fixed-recipe data-efficiency result artifacts."""

from pathlib import Path

from roofseg.data_efficiency import (
    collect_data_efficiency_results,
    plot_data_efficiency_histories,
    plot_data_efficiency_summary,
    summarize_data_efficiency_repetitions,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
results, paired, history, gains, overlap = collect_data_efficiency_results(ROOT, (1, 2, 3))
results.to_csv(ROOT / "reports/data_efficiency_results.csv", index=False)
paired.to_csv(ROOT / "reports/data_efficiency_paired.csv", index=False)
history.to_csv(ROOT / "reports/data_efficiency_history.csv", index=False)
gains.to_csv(ROOT / "reports/data_efficiency_size_gains.csv", index=False)
overlap.to_csv(ROOT / "reports/data_efficiency_subset_overlap.csv", index=False)
summary = summarize_data_efficiency_repetitions(results)
summary.to_csv(ROOT / "reports/data_efficiency_repetition_summary.csv", index=False)
figures = {
    "data_efficiency_summary.png": plot_data_efficiency_summary(results),
    "data_efficiency_histories_seed17.png": plot_data_efficiency_histories(history, 17),
    "data_efficiency_histories_seed29.png": plot_data_efficiency_histories(history, 29),
    "data_efficiency_histories_seed43.png": plot_data_efficiency_histories(history, 43),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
print("Data-efficiency results:\n", results.to_string(index=False))
print("\nPaired ImageNet-minus-random differences:\n", paired.to_string(index=False))
print("\nValidation-IoU gains between sizes:\n", gains.to_string(index=False))
print("\nShared training IDs:\n", overlap.to_string(index=False))
print("\nMean and sample SD across image selections:\n", summary.to_string(index=False))
