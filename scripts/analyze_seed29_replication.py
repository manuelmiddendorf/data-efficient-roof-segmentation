"""Create the matched Seed 17 and Seed 29 fixed-drop comparison artifacts."""

from pathlib import Path

import pandas as pd

from roofseg.late_training_analysis import (
    collect_subset_replication_comparison,
    plot_subset_replication_curves,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, paired, history, overlap_count = collect_subset_replication_comparison(ROOT)
results.to_csv(ROOT / "reports/optimization_subset_replication_results.csv", index=False)
paired.to_csv(ROOT / "reports/optimization_subset_replication_paired.csv", index=False)
history.to_csv(ROOT / "reports/optimization_subset_replication_history.csv", index=False)
pd.DataFrame([{
    "subset_seed_a": 17,
    "subset_seed_b": 29,
    "training_ids_each": 100,
    "shared_training_ids": overlap_count,
}]).to_csv(ROOT / "reports/optimization_subset_replication_overlap.csv", index=False)
figure = plot_subset_replication_curves(history)
figure.savefig(
    FIGURES / "optimization_subset_replication_curves.png",
    dpi=170,
    bbox_inches="tight",
)
print("Subset replication results:\n", results.to_string(index=False))
print("\nPaired ImageNet-minus-random differences:\n", paired.to_string(index=False))
print(f"\nShared training IDs: {overlap_count} of 100 per subset")
