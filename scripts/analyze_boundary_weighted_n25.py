"""Write compact tables and figures for the boundary-weighted BCE block."""

from pathlib import Path

import matplotlib.pyplot as plt

from roofseg.boundary_weighted import (
    REPLICATION_STRATEGY_REPETITIONS,
    choose_qualitative_examples,
    collect_boundary_results,
    plot_boundary_histories,
    plot_qualitative_examples,
)

ROOT = Path(__file__).resolve().parents[1]
results, effects, history, boundary_metrics = collect_boundary_results(
    ROOT, REPLICATION_STRATEGY_REPETITIONS
)
results.to_csv(ROOT / "reports/boundary_weighted_n25_results.csv", index=False)
effects.to_csv(ROOT / "reports/boundary_weighted_n25_effects.csv", index=False)
history.to_csv(ROOT / "reports/boundary_weighted_n25_history.csv", index=False)
boundary_metrics.to_csv(ROOT / "reports/boundary_weighted_n25_boundary_metrics.csv", index=False)
for metric, filename in (
    ("validation_mean_iou", "boundary_weighted_n25_iou.png"),
    ("validation_loss", "boundary_weighted_n25_loss.png"),
):
    figure = plot_boundary_histories(history, metric)
    figure.savefig(ROOT / "reports/figures" / filename, dpi=170, bbox_inches="tight")
    plt.close(figure)
selected = choose_qualitative_examples(boundary_metrics)
selected.to_csv(ROOT / "reports/boundary_weighted_n25_selected_examples.csv", index=False)
figure = plot_qualitative_examples(ROOT, selected)
figure.savefig(
    ROOT / "reports/figures/boundary_weighted_n25_examples.png", dpi=170, bbox_inches="tight"
)
plt.close(figure)
print(results.to_string(index=False))
print("\nPaired boundary-variant-minus-reference effects\n")
print(effects.to_string(index=False))
print("\nMetric-selected qualitative examples\n")
print(selected.to_string(index=False))
