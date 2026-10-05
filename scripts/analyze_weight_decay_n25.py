"""Create result artifacts for the n=25 strong-weight-decay block."""

from pathlib import Path

from roofseg.weight_decay import (
    collect_weight_decay_results,
    plot_weight_decay_histories,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
results, effects, history = collect_weight_decay_results(ROOT)
results.to_csv(ROOT / "reports/weight_decay_n25_results.csv", index=False)
effects.to_csv(ROOT / "reports/weight_decay_n25_effects.csv", index=False)
history.to_csv(ROOT / "reports/weight_decay_n25_history.csv", index=False)
for name, metric in (
    ("weight_decay_n25_iou.png", "validation_mean_iou"),
    ("weight_decay_n25_loss.png", "validation_loss"),
):
    figure = plot_weight_decay_histories(history, metric)
    figure.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
print("Weight-decay results:\n", results.to_string(index=False))
print("\nStrong-minus-reference changes:\n", effects.to_string(index=False))
