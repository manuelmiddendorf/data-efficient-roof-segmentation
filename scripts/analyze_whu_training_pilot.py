"""Create compact WHU pilot tables and figures from completed artifacts."""

from pathlib import Path

import torch

from roofseg.whu_pilot_analysis import (
    choose_whu_validation_examples, collect_whu_pilot_tables,
    plot_cross_dataset_effects, plot_whu_learning_curves,
    plot_whu_validation_predictions, rid_whu_initialization_effects,
    whu_initialization_effects,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

results, history, metrics = collect_whu_pilot_tables(ROOT)
examples = choose_whu_validation_examples(metrics)
effects, effect_summary = whu_initialization_effects(results)
cross_dataset = rid_whu_initialization_effects(ROOT, effects)
results.to_csv(ROOT / "reports/whu_training_pilot_results.csv", index=False)
history.to_csv(ROOT / "reports/whu_training_pilot_history.csv", index=False)
examples.to_csv(ROOT / "reports/whu_training_pilot_examples.csv", index=False)
effects.to_csv(ROOT / "reports/whu_training_pilot_effects.csv", index=False)
effect_summary.to_csv(ROOT / "reports/whu_training_pilot_effect_summary.csv", index=False)
cross_dataset.to_csv(ROOT / "reports/whu_rid_initialization_effects.csv", index=False)

figures = {
    "07_whu_pilot_learning_curves.png": plot_whu_learning_curves(history),
    "07_whu_rid_initialization_effects.png": plot_cross_dataset_effects(cross_dataset),
    "07_whu_pilot_validation_examples.png": plot_whu_validation_predictions(ROOT, examples),
}
for name, figure in figures.items():
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")

print(results.to_string(index=False))
print("\nWHU paired effects:\n", effects.to_string(index=False))
print("\nWHU effect summary:\n", effect_summary.to_string(index=False))
print("\nRID and WHU effects:\n", cross_dataset.to_string(index=False))
print("\nSelected common examples:\n", examples.to_string(index=False))
