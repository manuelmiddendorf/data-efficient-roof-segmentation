"""Build descriptive final-test plots and the frozen qualitative examples."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.training import restore_checkpoint
from roofseg.training_data import build_tensor_store

ROOT = Path(__file__).resolve().parents[1]
results = pd.read_csv(ROOT / 'reports/final_test_results.csv')
summary = pd.read_csv(ROOT / 'reports/final_test_summary.csv')
primary = summary[(summary.analysis == 'primary_learning_curve') & (summary.population == 'primary')]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, dataset in zip(axes, ('rid', 'whu_aerial')):
    data = primary[primary.dataset == dataset].sort_values('training_size')
    for init, label, color in [('random', 'Random', '#b85c38'), ('imagenet', 'ImageNet', '#247ba0')]:
        ax.errorbar(data.training_size, data[init + '_mean'], yerr=data[init + '_std'],
                    marker='o', capsize=4, color=color, label=label)
    ax.set(xlabel='Labelled training images', title=dataset.upper(), xticks=[25, 100, 500], ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend()
axes[0].set_ylabel('Mean per-image test IoU')
fig.suptitle('Validation-selected checkpoints; mean ± subset-selection SD')
fig.tight_layout()
(ROOT / 'reports/figures').mkdir(exist_ok=True)
fig.savefig(ROOT / 'reports/figures/08_final_test_learning_curves.png', dpi=170)
plt.close(fig)
# Keep source-image panels local; public redistribution permission is not established.
local = ROOT / '.local/final_test_figures'
local.mkdir(parents=True, exist_ok=True)
per_image = pd.read_csv(ROOT / 'reports/final_test_per_image.csv', dtype={'id': str})
matrix = pd.read_csv(ROOT / 'reports/final_evaluation_checkpoints.csv')
selected = []
for dataset in ('rid', 'whu_aerial'):
    entries = matrix[(matrix.analysis == 'primary_learning_curve') & (matrix.dataset == dataset)
                     & (matrix.training_size == 100) & (matrix.subset_seed == 17)]
    tables = {}
    for index, entry in entries.iterrows():
        tables[entry.initialization] = per_image[per_image.entry_index == index].set_index('id')
    ids = tables['random'].index.tolist()
    if dataset == 'rid':
        eligibility = pd.read_csv(ROOT / 'reports/final_evaluation_rid_test_ids.csv', dtype={'id': str})
        allowed = set(eligibility.loc[eligibility.primary_test, 'id'])
        ids = [i for i in ids if i in allowed]
    positive = [i for i in ids if tables['random'].loc[i, 'tp'] + tables['random'].loc[i, 'fn'] > 0]
    empty = [i for i in ids if i not in positive]
    mean_iou = {i: float(np.mean([tables[k].loc[i, 'iou'] for k in tables])) for i in positive}
    ranked = sorted(positive, key=lambda i: mean_iou[i]) # stable saved-ID order breaks ties
    choices = [('low paired IoU', i) for i in ranked[:2]]
    choices += [('high paired IoU', i) for i in sorted(positive, key=lambda i: -mean_iou[i])[:2]]
    if positive:
        median = np.median(list(mean_iou.values()))
        choices.append(('near median paired IoU', min(positive, key=lambda i: abs(mean_iou[i] - median))))
    if empty:
        worst = max(empty, key=lambda i: np.mean([tables[k].loc[i, 'fp'] for k in tables]))
        choices.append(('highest empty-target foreground', worst))
    manifest_path = 'data/metadata/data_manifest.json' if dataset == 'rid' else 'data/metadata/whu/data_manifest.json'
    store = build_tensor_store(ROOT, load_json(ROOT / manifest_path), 'rid' if dataset == 'rid' else 'whu_aerial')
    examples = [i for _, i in choices]
    predictions = {}
    torch.set_num_threads(4)
    for _, entry in entries.iterrows():
        config = load_json(ROOT / entry.run_directory / 'config.json')
        model, _ = build_model('random', config['seeds']['model'])
        restore_checkpoint(model, ROOT / entry.run_directory / entry.checkpoint_filename, entry.config_sha256, entry.input_sha256)
        model.to('mps').eval()
        with torch.inference_mode():
            predictions[entry.initialization] = torch.cat([
                model(store.batch(examples[start:start+4])['image'].to('mps')).sigmoid().cpu()
                for start in range(0, len(examples), 4)])[:, 0].numpy() >= .5
        del model
        torch.mps.empty_cache()
    fig, axes = plt.subplots(len(choices), 4, figsize=(12, 3 * len(choices)), squeeze=False)
    for row, (reason, sample_id) in enumerate(choices):
        image, target = store.load_raw(sample_id)
        panels = [image.permute(1, 2, 0).numpy(), target[0].numpy(), predictions['random'][row], predictions['imagenet'][row]]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, cmap=None if column == 0 else 'gray', vmin=0, vmax=1)
            axes[row, column].axis('off')
        axes[row, 0].set_title(f'{sample_id}: {reason}', fontsize=9)
        axes[row, 1].set_title('Reference (white = foreground)')
        for column, init in ((2, 'random'), (3, 'imagenet')):
            axes[row, column].set_title(f'{init}: IoU {tables[init].loc[sample_id, "iou"]:.3f}')
        selected.append({'dataset': dataset, 'id': sample_id, 'selection_reason': reason})
    fig.suptitle(f'{dataset.upper()}: metric-selected examples, n=100, subset seed 17')
    fig.tight_layout()
    fig.savefig(local / f'{dataset}_examples.png', dpi=130)
    plt.close(fig)
pd.DataFrame(selected).to_csv(ROOT / 'reports/final_test_qualitative_ids.csv', index=False)
print(primary.to_string(index=False))
print('Qualitative source-image panels are local under .local/final_test_figures/.')
