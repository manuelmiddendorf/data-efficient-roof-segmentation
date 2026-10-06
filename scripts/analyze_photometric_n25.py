"""Write compact tables and learning curves for the n=25 colour block."""

from pathlib import Path

import matplotlib.pyplot as plt

from roofseg.photometric_augmentation import collect_photometric_results, plot_photometric_histories

ROOT = Path(__file__).resolve().parents[1]
results, effects, history = collect_photometric_results(ROOT)
results.to_csv(ROOT / "reports/photometric_n25_results.csv", index=False)
effects.to_csv(ROOT / "reports/photometric_n25_effects.csv", index=False)
history.to_csv(ROOT / "reports/photometric_n25_history.csv", index=False)
for metric, filename in (
    ("validation_mean_iou", "photometric_n25_iou.png"),
    ("validation_loss", "photometric_n25_loss.png"),
):
    figure = plot_photometric_histories(history, metric)
    figure.savefig(ROOT / "reports/figures" / filename, dpi=170, bbox_inches="tight")
    plt.close(figure)
print(results.to_string(index=False))
print("\nPaired augmented-minus-reference effects\n")
print(effects.to_string(index=False))
