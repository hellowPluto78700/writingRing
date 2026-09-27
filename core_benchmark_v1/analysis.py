"""Read-only notebook helpers. No training, probes, or subprocess execution."""
from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def results_root(repo: Path) -> Path:
    return Path(os.environ.get('CORE_BENCHMARK_RESULTS', str(repo / 'core_benchmark_v1/results/main'))).expanduser().resolve()


def load_manifest(root: Path) -> dict:
    path = root / 'aggregate/manifest.json'
    if not path.exists():
        print(f'No finalized results at {root}. Run prepare -> arrays -> finalize first. No numbers are inferred here.')
        return {}
    manifest = json.loads(path.read_text())
    lock = json.loads((root / 'protocol.lock.json').read_text())
    if manifest.get('status') != 'PASS' or manifest.get('identity') != lock.get('identity'):
        raise ValueError('Aggregate manifest is incomplete or belongs to another protocol')
    if manifest['profile'] != 'production':
        print('SYNTHETIC SMOKE DATA: execution validation only; not scientific performance results.')
    return manifest


def table(root: Path, name: str, block: str | None = None) -> pd.DataFrame:
    directory = root / 'aggregate'
    if not (directory / 'manifest.json').exists():
        return pd.DataFrame()
    if block is not None:
        directory = directory / block
    path = directory / f'{name}.csv'
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def plot_native(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(frame))
    for split in ('train', 'val', 'test'):
        ax.errorbar(x, 100 * frame[f'{split}_ba_mean'], yerr=100 * frame[f'{split}_ba_std'].fillna(0), marker='o', capsize=3, label=split)
    ax.set_xticks(x, frame['display_case'] if 'display_case' in frame else frame['case'])
    ax.set(ylabel='Balanced accuracy (%)', title=title, ylim=(0, 102))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_probe(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        return
    selected = frame[(frame.state == 'spike') & (frame.decoder == 'no_bias') & (frame.aggregation.isin(['whole_count', 'fixed250_ordered', 'fixed250_shuffled']))]
    if selected.empty:
        return
    selected = selected.copy()
    selected['case_layer'] = selected['case'] + '/' + selected['layer']
    pivot = selected.pivot(index='case_layer', columns='aggregation', values='test_ba_mean')
    fig, ax = plt.subplots(figsize=(max(9, len(pivot) * 0.65), 4))
    for name in pivot:
        ax.plot(np.arange(len(pivot)), 100 * pivot[name], marker='o', label=name)
    ax.set_xticks(np.arange(len(pivot)), pivot.index, rotation=45, ha='right')
    ax.set(ylabel='Test balanced accuracy (%)', title=title, ylim=(0, 102))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_readout(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(frame))
    ax.errorbar(x, 100 * frame.test_ba_mean, yerr=100 * frame.test_ba_std.fillna(0), marker='o', linestyle='none', capsize=3)
    ax.set_xticks(x, frame['case'] + '/' + frame['mode'], rotation=35, ha='right')
    ax.set(ylabel='Test balanced accuracy (%)', title='Readout-only conversion and W adaptation', ylim=(0, 102))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show()
