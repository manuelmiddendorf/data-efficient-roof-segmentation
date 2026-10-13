"""Frozen final-test inference and equal-image descriptive summaries."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .integrity import load_json, sha256_file
from .model import build_model
from .objectives import average_precision, binary_metrics
from .run_artifacts import json_digest, selected_input_digest, write_csv, write_json
from .training import restore_checkpoint
from .training_data import build_tensor_store


def summarize_images(rows: list[dict], pixels_per_image: int = 512 * 512) -> dict:
    """Aggregate equal-image metrics, with null values for absent groups.

    Parameters
    ----------
    rows : list of dict
        Canonical confusion counts, IoU, Dice and positive-target AP per image.
    pixels_per_image : int
        Native image area, used only for empty-target foreground fractions.
    """
    if not rows:
        raise ValueError("An evaluation population must be nonempty.")
    def mean(group, key):
        values = [r[key] for r in group if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    positive = [r for r in rows if r['tp'] + r['fn'] > 0]
    empty = [r for r in rows if r['tp'] + r['fn'] == 0]
    return {
        'image_count': len(rows), 'mean_iou': mean(rows, 'iou'),
        'mean_dice': mean(rows, 'dice'),
        'mean_average_precision': mean(positive, 'average_precision'),
        'ap_image_count': sum(r.get('average_precision') is not None for r in positive),
        'positive_image_count': len(positive), 'positive_mean_iou': mean(positive, 'iou'),
        'positive_mean_dice': mean(positive, 'dice'),
        'empty_image_count': len(empty),
        'empty_mean_foreground_fraction': float(np.mean([r['fp'] / pixels_per_image for r in empty])) if empty else None,
        'empty_fraction_with_any_prediction': float(np.mean([r['fp'] > 0 for r in empty])) if empty else None,
    }


def evaluate_test(model, store, ids: list[str], device: torch.device) -> list[dict]:
    """Predict native images in fixed batches of four and evaluate at 0.5.

    Store inputs are normalized float32 RGB and aligned binary 512-square targets.
    No augmentation or training occurs; AP uses the canonical tie-aware metric.
    """
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids), 4):
            batch = store.batch(ids[start:start + 4])
            probabilities = model(batch['image'].to(device)).sigmoid().cpu().numpy()[:, 0]
            targets = batch['target'].numpy()[:, 0]
            if not np.isfinite(probabilities).all():
                raise ValueError('Non-finite test probabilities.')
            for sample_id, probability, target in zip(batch['id'], probabilities, targets):
                rows.append({'id': sample_id, **binary_metrics(probability, target),
                             'average_precision': average_precision(probability, target)})
    return rows


def verify_frozen_inputs(root: Path):
    """Verify protocol hashes, checkpoint identities and actual test file bytes.

    Returns the frozen matrix, manifests, ordered populations and execution identity.
    """
    review = load_json(root / 'reports/final_review_checks.json')
    frozen = {}
    for name, record in review['frozen_protocol_files'].items():
        path = root / (name if name.startswith('docs/') else 'reports/' + name)
        actual = sha256_file(path)
        if actual != record['sha256']:
            raise ValueError(f'Frozen protocol mismatch: {name}')
        frozen[name] = actual
    matrix = pd.read_csv(root / 'reports/final_evaluation_checkpoints.csv', keep_default_na=False)
    if len(matrix) != 62 or matrix['run_directory'].nunique() != 52:
        raise ValueError('Unexpected frozen checkpoint matrix.')
    manifests = {d: load_json(root / p) for d, p in {
        'rid': 'data/metadata/data_manifest.json', 'whu': 'data/metadata/whu/data_manifest.json'}.items()}
    eligibility = pd.read_csv(root / 'reports/final_evaluation_rid_test_ids.csv', dtype={'id': str})
    rid_ids = eligibility['id'].tolist()
    primary_ids = eligibility.loc[eligibility['primary_test'], 'id'].tolist()
    splits = {d: load_json(root / p) for d, p in {
        'rid': 'data/metadata/splits.json', 'whu': 'data/metadata/whu/splits.json'}.items()}
    if rid_ids != splits['rid']['roles']['test'] or len(primary_ids) != 254:
        raise ValueError('RID eligibility differs from frozen population.')
    populations = {'rid': rid_ids, 'whu': splits['whu']['roles']['test']}
    references = review['test_input_identities_from_existing_manifest']
    for dataset, ids in populations.items():
        expected = references['rid_full_region' if dataset == 'rid' else 'whu_provider_test']
        if selected_input_digest(manifests[dataset], ids) != expected:
            raise ValueError(f'Test input identity differs: {dataset}')
        records = {r['id']: r for r in manifests[dataset]['samples']}
        for sample_id in ids:
            record = records[sample_id]
            for kind in ('image', 'mask'):
                if sha256_file(root / record[kind]) != record[kind + '_sha256']:
                    raise ValueError(f'Test file mismatch: {dataset}/{sample_id}/{kind}')
    if selected_input_digest(manifests['rid'], primary_ids) != references['rid_primary']:
        raise ValueError('Primary RID input identity differs.')
    for entry in matrix.to_dict('records'):
        directory = root / entry['run_directory']
        config = load_json(directory / 'config.json')
        if json_digest(config) != entry['config_sha256']:
            raise ValueError(f'Config mismatch: {directory}')
        identity = load_json(directory / 'metadata.json')['compatibility_identity']
        inputs = {k: v for k, v in identity.items() if k not in ('input_sha256', 'config_sha256')}
        if json_digest(inputs) != entry['input_sha256'] or identity['input_sha256'] != entry['input_sha256']:
            raise ValueError(f'Run input identity mismatch: {directory}')
        for role in ('training', 'validation'):
            if selected_input_digest(manifests[entry['dataset']], config[role + '_ids']) != identity[role + '_inputs_sha256']:
                raise ValueError(f'Run {role} input mismatch: {directory}')
        if set(config['training_ids'] + config['validation_ids']) & set(populations[entry['dataset']]):
            raise ValueError('Active test IDs used in training/validation.')
        if config['probability_threshold'] != 0.5 or config['batch_size'] != 4:
            raise ValueError('Unexpected inference configuration.')
        if sha256_file(directory / entry['checkpoint_filename']) != entry['checkpoint_sha256']:
            raise ValueError(f'Checkpoint mismatch: {directory}')
    return matrix, manifests, populations, primary_ids, frozen


def run_final_evaluation(root: Path, device_name: str = 'mps') -> None:
    """Execute all 62 frozen entries, safely reusing compatible completed outputs.

    Per-entry outputs live in ignored runs/final_evaluation; small report tables
    and one execution record are exported only after every entry is complete.
    """
    if device_name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS is unavailable; no fallback allowed.')
    device = torch.device(device_name)
    torch.set_num_threads(4)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Commit source and protocol before accessing test data.')
    matrix, manifests, populations, primary_ids, frozen = verify_frozen_inputs(root)
    execution = {'code_commit': commit, 'device': device_name, 'torch': torch.__version__,
                 'python': platform.python_version(), 'platform': platform.platform(),
                 'frozen_protocol': frozen, 'threshold': 0.5, 'batch_size': 4,
                 'test_inputs': {d: selected_input_digest(manifests[d], ids) for d, ids in populations.items()}}
    base = root / 'runs/final_evaluation'
    base.mkdir(parents=True, exist_ok=True)
    aggregate, all_rows = [], []
    started = time.perf_counter()
    for index, entry in enumerate(matrix.to_dict('records')):
        directory = base / f'entry_{index:02d}'
        directory.mkdir(exist_ok=True)
        identity = {'entry': entry, 'execution': execution}
        config_path, metrics_path, summary_path = (directory / name for name in ('config.json', 'metrics.csv', 'summary.json'))
        if config_path.exists() and load_json(config_path) != identity:
            raise ValueError(f'Conflicting evaluation: {directory}')
        if not config_path.exists():
            write_json(config_path, identity)
            write_json(directory / 'metadata.json', execution)
        if summary_path.exists():
            saved = load_json(summary_path)
            if sha256_file(metrics_path) != saved['metrics_sha256']:
                raise ValueError('Completed evaluation metrics changed.')
            rows = pd.read_csv(metrics_path, dtype={'id': str}).replace({np.nan: None}).to_dict('records')
            if [r['id'] for r in rows] != populations[entry['dataset']]:
                raise ValueError('Completed evaluation has incorrect test order.')
            print(f'{index + 1}/62 reused {entry["dataset"]} n={entry["training_size"]} {entry["initialization"]}', flush=True)
        else:
            config = load_json(root / entry['run_directory'] / 'config.json')
            # Initial random weights are entirely replaced, avoiding a download at inference.
            model, _ = build_model('random', config['seeds']['model'])
            step = restore_checkpoint(model, root / entry['run_directory'] / entry['checkpoint_filename'],
                                      entry['config_sha256'], entry['input_sha256'])
            if step != entry['selected_update']:
                raise ValueError('Frozen selected update differs from checkpoint.')
            model.to(device)
            store = build_tensor_store(root, manifests[entry['dataset']], 'rid' if entry['dataset'] == 'rid' else 'whu_aerial')
            elapsed = time.perf_counter()
            rows = evaluate_test(model, store, populations[entry['dataset']], device)
            write_csv(metrics_path, rows)
            write_json(summary_path, {'completed': True, 'seconds': time.perf_counter() - elapsed,
                                     'metrics_sha256': sha256_file(metrics_path), **summarize_images(rows)})
            del model
            if device_name == 'mps':
                torch.mps.empty_cache()
            print(f'{index + 1}/62 evaluated {entry["dataset"]} n={entry["training_size"]} {entry["initialization"]}: IoU={summarize_images(rows)["mean_iou"]:.4f}', flush=True)
        groups = {'primary': rows}
        if entry['dataset'] == 'rid':
            primary_set = set(primary_ids)
            groups = {'primary': [r for r in rows if r['id'] in primary_set], 'full_region_sensitivity': rows}
        for population, group in groups.items():
            aggregate.append({**entry, 'population': population, **summarize_images(group)})
        all_rows.extend({'entry_index': index, 'dataset': entry['dataset'], **r} for r in rows)
    results = pd.DataFrame(aggregate)
    results.to_csv(root / 'reports/final_test_results.csv', index=False)
    pd.DataFrame(all_rows).to_csv(root / 'reports/final_test_per_image.csv', index=False)
    keys = ['analysis', 'dataset', 'training_size', 'subset_seed', 'selection_strategy', 'selection_through_update', 'population']
    paired = results.pivot(index=keys, columns='initialization', values='mean_iou').reset_index()
    paired['imagenet_minus_random'] = paired['imagenet'] - paired['random']
    paired.to_csv(root / 'reports/final_test_paired.csv', index=False)
    grouping = [k for k in keys if k != 'subset_seed']
    description = paired.groupby(grouping)[['random', 'imagenet', 'imagenet_minus_random']].agg(['mean', 'std', 'count'])
    description.columns = ['_'.join(column) for column in description.columns]
    description.reset_index().to_csv(root / 'reports/final_test_summary.csv', index=False)
    write_json(root / 'reports/final_test_execution.json', {**execution, 'completed': True,
        'checkpoint_entries': len(matrix), 'elapsed_seconds_this_invocation': time.perf_counter() - started,
        'artifacts': {name: sha256_file(root / 'reports' / name) for name in (
            'final_test_results.csv', 'final_test_per_image.csv', 'final_test_paired.csv', 'final_test_summary.csv')}})
