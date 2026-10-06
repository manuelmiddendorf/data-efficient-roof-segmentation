"""Write compact tables and learning curves for the n=25 BatchNorm block."""

from pathlib import Path

import matplotlib.pyplot as plt

from roofseg.batch_norm import collect_batch_norm_results, plot_batch_norm_histories

ROOT = Path(__file__).resolve().parents[1]
results, effects, history = collect_batch_norm_results(ROOT)
results.to_csv(ROOT / "reports/encoder_batch_norm_n25_results.csv", index=False)
effects.to_csv(ROOT / "reports/encoder_batch_norm_n25_effects.csv", index=False)
history.to_csv(ROOT / "reports/encoder_batch_norm_n25_history.csv", index=False)
for metric, filename in (
    ("validation_mean_iou", "encoder_batch_norm_n25_iou.png"),
    ("validation_loss", "encoder_batch_norm_n25_loss.png"),
):
    figure = plot_batch_norm_histories(history, metric)
    figure.savefig(ROOT / "reports/figures" / filename, dpi=170, bbox_inches="tight")
    plt.close(figure)
print(results.to_string(index=False))
print("\nPaired frozen-minus-updated effects\n")
print(effects.to_string(index=False))
