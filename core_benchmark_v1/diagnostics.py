"""Descriptive lag stability and causal state-reset/common-suffix replay controls."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .data import loader
from .model import BenchmarkNet
from .protocol import Protocol, Run, SPLITS
from .storage import save_json
from .training import metrics


def vector_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (na > 0) & (nb > 0)
    ca, cb = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    nca, ncb = np.linalg.norm(ca, axis=1), np.linalg.norm(cb, axis=1)
    corr_valid = (nca > 0) & (ncb > 0)
    def average(values: np.ndarray) -> float | None:
        return float(values.mean()) if len(values) else None
    return {'pair_count': len(a), 'cosine_pair_count': int(valid.sum()), 'corr_pair_count': int(corr_valid.sum()),
            'cosine': average((a[valid] * b[valid]).sum(1) / (na[valid] * nb[valid])),
            'correlation': average((ca[corr_valid] * cb[corr_valid]).sum(1) / (nca[corr_valid] * ncb[corr_valid])),
            'normalized_l2': average(np.linalg.norm(a[valid] - b[valid], axis=1) / (0.5 * (na[valid] + nb[valid])))}


def lag_rows(traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], run: Run, p: Protocol) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        lengths = arrays[f'{split}_lengths']
        for li in range(len(run.shifts)):
            for state in p.probe_states:
                z = traces[f'{split}__L{li+1}__{state}']
                for lag in p.lags:
                    if lag >= p.steps:
                        continue
                    valid = np.arange(p.steps - lag)[None, :] + lag < lengths[:, None]
                    row = {'case': run.case, 'seed': run.seed, 'split': split, 'layer': f'L{li+1}', 'state': state,
                           'lag': lag, 'lag_ms': 1000 * lag / p.fs}
                    rows.append({**row, **vector_similarity(z[:, :-lag][valid], z[:, lag:][valid])})
    return rows


@torch.no_grad()
def history_rows(model: BenchmarkNet, traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], run: Run, p: Protocol) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    interventions = [(f'L{i+1}', (i,)) for i in range(len(run.shifts))] + [('all', tuple(range(len(run.shifts))))]
    for split in ('val', 'test'):
        lengths_all = arrays[f'{split}_lengths']
        y_all = arrays[f'{split}_y']
        full = traces[f'{split}__evidence']
        for requested_ms in p.history_ms:
            history = max(1, int(round(requested_ms * p.fs / 1000)))
            eligible = lengths_all > history
            if not eligible.any():
                continue
            start_all = lengths_all - history
            suffix = np.arange(p.steps)[None, :] >= start_all[:, None]
            full_suffix_scores = (full * suffix[:, :, None]).sum(1)
            full_scores = full.sum(1)
            for name, selected_layers in interventions:
                changed = []
                for x, _, lengths in loader(arrays, split, p, run.seed):
                    boundaries = lengths - history
                    changed.append(model(x, lengths, reset_at=boundaries, reset_layers=selected_layers)['evidence'].numpy())
                evidence = np.concatenate(changed)
                changed_scores = evidence.sum(1)
                changed_suffix_scores = (evidence * suffix[:, :, None]).sum(1)
                y = y_all[eligible]
                original_metrics = metrics(y, full_scores[eligible].argmax(1))
                suffix_metrics = metrics(y, full_suffix_scores[eligible].argmax(1))
                changed_metrics = metrics(y, changed_scores[eligible].argmax(1))
                changed_suffix_metrics = metrics(y, changed_suffix_scores[eligible].argmax(1))
                rows.append({'case': run.case, 'seed': run.seed, 'split': split, 'reset_layers': name,
                    'requested_history_ms': requested_ms, 'history_steps': history, 'actual_history_ms': history * 1000 / p.fs,
                    'eligible_samples': int(eligible.sum()), 'classes_present': int(len(np.unique(y))),
                    'full_ba': original_metrics['ba'], 'reset_full_ba': changed_metrics['ba'],
                    'full_suffix_ba': suffix_metrics['ba'], 'reset_suffix_ba': changed_suffix_metrics['ba'],
                    'full_ba_drop': original_metrics['ba'] - changed_metrics['ba'],
                    'suffix_ba_drop': suffix_metrics['ba'] - changed_suffix_metrics['ba'],
                    'suffix_score_change_l2': float(np.linalg.norm(changed_suffix_scores[eligible] - full_suffix_scores[eligible], axis=1).mean())})
    return rows


def run_diagnostics(directory: Path, model: BenchmarkNet, traces: dict[str, np.ndarray], arrays: dict[str, np.ndarray], run: Run, p: Protocol) -> None:
    enabled = p.diagnostics and run.case in p.diagnostic_cases
    save_json(directory / 'diagnostics.json', {'enabled': enabled,
        'lag': lag_rows(traces, arrays, run, p) if enabled else [],
        'history': history_rows(model, traces, arrays, run, p) if enabled else [],
        'lag_support': 'all valid timestep pairs in all samples; undefined zero/constant-vector similarities excluded and counted',
        'history_support': 'one endpoint per sample: final valid timestep; common suffix unchanged; reset both synaptic and membrane state',
        'scope': 'conditional history dependence and stability, not a proof of high-level abstraction'})
