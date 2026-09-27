"""Matched temporal probes; train-only scaling and validation-only C selection."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .protocol import Protocol, Run, SPLITS, paired_seed
from .storage import save_json, save_npz
from .training import metrics

AGGREGATIONS = ('whole_count', 'fixed250_ordered', 'fixed250_shuffled', 'relative10_ordered', 'relative10_shuffled')
DECODERS = ('no_bias', 'affine')


def probe_specs(p: Protocol) -> list[tuple[str, str, int]]:
    return [(aggregation, decoder, seed)
            for aggregation in AGGREGATIONS
            for seed in (p.shuffle_seeds if aggregation.endswith('shuffled') else (-1,))
            for decoder in DECODERS]


def temporal_features(z: np.ndarray, lengths: np.ndarray, sample_ids: np.ndarray, aggregation: str, p: Protocol, shuffle_seed: int = -1) -> np.ndarray:
    if z.ndim != 3 or z.shape[:2] != (len(lengths), p.steps) or len(sample_ids) != len(lengths):
        raise ValueError('Unaligned temporal representation')
    values = z.astype(np.float32, copy=True)
    values[np.arange(p.steps)[None, :] >= lengths[:, None]] = 0
    if aggregation == 'whole_count':
        return values.sum(1)
    if aggregation.startswith('fixed250'):
        bins = values.reshape(len(values), p.steps // p.fixed_steps, p.fixed_steps, values.shape[2]).sum(2)
        movable = lengths // p.fixed_steps
    elif aggregation.startswith('relative10'):
        bins = np.zeros((len(values), p.relative_bins, values.shape[2]), dtype=np.float32)
        for i, length in enumerate(lengths):
            boundaries = np.linspace(0, int(length), p.relative_bins + 1).astype(int)
            for b in range(p.relative_bins):
                bins[i, b] = values[i, boundaries[b]:boundaries[b+1]].sum(0)
        movable = np.full(len(values), p.relative_bins, dtype=int)
    else:
        raise ValueError(aggregation)
    if aggregation.endswith('shuffled'):
        if shuffle_seed < 0:
            raise ValueError('Shuffled probes require an explicit shuffle replicate seed')
        for i, count in enumerate(movable):
            # Per-sample permutation, independent of model seed, state, layer and label.
            rng = np.random.default_rng(paired_seed(shuffle_seed, f'{aggregation}:{sample_ids[i]}'))
            permutation = rng.permutation(int(count))
            # Fixed250: preserve the partial last bin and padding in their positions.
            bins[i, :count] = bins[i, permutation].copy()
    return bins.reshape(len(values), -1)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, decoder: str, p: Protocol) -> tuple[StandardScaler, LogisticRegression, list[dict[str, Any]]]:
    if decoder not in DECODERS:
        raise ValueError(decoder)
    # Both geometries have identical scale-only preprocessing; no hidden intercept.
    scaler = StandardScaler(with_mean=False, with_std=True)
    train_scaled = scaler.fit_transform(x_train.astype(np.float64))
    val_scaled = scaler.transform(x_val.astype(np.float64))
    best: LogisticRegression | None = None
    best_ba = -1.0
    search: list[dict[str, Any]] = []
    for c in sorted(p.c_grid):
        candidate = LogisticRegression(C=c, fit_intercept=decoder == 'affine', solver='lbfgs',
                                       max_iter=p.probe_max_iter, tol=p.probe_tol, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            candidate.fit(train_scaled, y_train)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        val_ba = metrics(y_val, candidate.predict(val_scaled))['ba']
        search.append({'C': c, 'val_ba': val_ba, 'converged': converged, 'iterations': int(candidate.n_iter_.max())})
        # Ties retain the smaller C. No test data are available inside this function.
        if converged and val_ba > best_ba + 1e-12:
            best, best_ba = candidate, val_ba
    if best is None:
        raise RuntimeError(f'No converged {decoder} probe: {search}')
    if decoder == 'no_bias' and np.any(best.intercept_ != 0):
        raise AssertionError('Unexpected no-bias intercept')
    return scaler, best, search


def run_probes(directory: Path, traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], run: Run, p: Protocol) -> list[dict[str, Any]]:
    rows, search_rows = [], []
    decoders: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for split in SPLITS:
        predictions[f'{split}__ids'] = arrays[f'{split}_ids']
        predictions[f'{split}__y'] = arrays[f'{split}_y']
    for li in range(len(run.shifts)):
        layer = f'L{li+1}'
        for state in p.probe_states:
            for aggregation in AGGREGATIONS:
                seeds = p.shuffle_seeds if aggregation.endswith('shuffled') else (-1,)
                for shuffle_seed in seeds:
                    features = {s: temporal_features(traces[f'{s}__{layer}__{state}'], arrays[f'{s}_lengths'], arrays[f'{s}_ids'], aggregation, p, shuffle_seed) for s in SPLITS}
                    for decoder in DECODERS:
                        scaler, model, search = fit_probe(features['train'], arrays['train_y'], features['val'], arrays['val_y'], decoder, p)
                        key = f'{layer}__{state}__{aggregation}__{decoder}__shuffle{shuffle_seed}'
                        row: dict[str, Any] = {'run_key': run.key, 'case': run.case, 'seed': run.seed, 'block': run.block,
                            'layer': layer, 'state': state, 'aggregation': aggregation, 'decoder': decoder,
                            'shuffle_seed': shuffle_seed, 'C': float(model.C), 'dimension': features['train'].shape[1],
                            'converged': True, 'with_mean': False, 'n_shuffle_replicates': len(seeds)}
                        for split in SPLITS:
                            prediction = model.predict(scaler.transform(features[split].astype(np.float64)))
                            row.update({f'{split}_{name}': value for name, value in metrics(arrays[f'{split}_y'], prediction).items()})
                            predictions[f'{key}__{split}'] = prediction
                        row['train_test_gap'] = row['train_ba'] - row['test_ba']
                        rows.append(row)
                        search_rows.extend({'probe_key': key, **candidate} for candidate in search)
                        for name, value in (('coef', model.coef_), ('intercept', model.intercept_), ('scale', scaler.scale_), ('classes', model.classes_)):
                            decoders[f'{key}__{name}'] = np.asarray(value)
            print(f'{run.key} probes completed: {layer}/{state}', flush=True)
    expected = len(run.shifts) * len(p.probe_states) * len(probe_specs(p))
    if len(rows) != expected:
        raise AssertionError(f'Expected {expected} probe rows, got {len(rows)}')
    save_json(directory / 'probes.json', {'rows': rows, 'expected_rows': expected})
    save_json(directory / 'probe_search.json', {'rows': search_rows})
    save_npz(directory / 'probe_decoders.npz', decoders)
    save_npz(directory / 'probe_predictions.npz', predictions)
    return rows
