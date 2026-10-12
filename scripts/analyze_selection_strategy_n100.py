"""Create reports for the random-versus-spatial n=100 comparison."""
from pathlib import Path
import matplotlib.pyplot as plt
from roofseg.integrity import load_json
from roofseg.selection_strategy import (choose_qualitative_examples, collect_selection_results,
    plot_qualitative_examples, plot_selection_map, plot_selection_performance,
    spatial_subset_diagnostics, subset_id_overlaps)
ROOT = Path(__file__).resolve().parents[1]
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
figures = ROOT / "reports/figures"; figures.mkdir(parents=True, exist_ok=True)
spatial = spatial_subset_diagnostics(manifest, splits)
overlap = subset_id_overlaps(splits)
results, effects, summary = collect_selection_results(ROOT)
selected = choose_qualitative_examples(ROOT)
for frame, name in ((spatial,"selection_strategy_n100_spatial_diagnostics.csv"),
                    (overlap,"selection_strategy_n100_id_overlap.csv"),
                    (results,"selection_strategy_n100_results.csv"),
                    (effects,"selection_strategy_n100_effects.csv"),
                    (summary,"selection_strategy_n100_effect_summary.csv"),
                    (selected,"selection_strategy_n100_selected_examples.csv")):
    frame.to_csv(ROOT / "reports" / name, index=False)
for figure, name in ((plot_selection_map(manifest, splits),"selection_strategy_n100_map.png"),
                     (plot_selection_performance(results,effects),"selection_strategy_n100_performance.png"),
                     (plot_qualitative_examples(ROOT,selected),"selection_strategy_n100_examples.png")):
    figure.savefig(figures / name, dpi=170, bbox_inches="tight"); plt.close(figure)
print("Results\n", results.to_string(index=False))
print("\nSpatial minus random effects\n", effects.to_string(index=False))
print("\nEffect summary\n", summary.to_string(index=False))
print("\nSpatial diagnostics\n", spatial.to_string(index=False))
print("\nSelected examples\n", selected.to_string(index=False))
